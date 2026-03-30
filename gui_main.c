/**
 * @file gui_main.c
 * @brief SDL2 GUI for the OCR crossword solver.
 *
 * Usage: ./gui [--model <path>]
 *
 * Features:
 *   - Open crossword image via file dialog
 *   - Load CNN model (auto-detects latest in models/ by default)
 *   - Enter words to search (comma-separated)
 *   - Run OCR pipeline and display found words highlighted in red
 *   - Original image pixels are never modified
 *
 * Dependencies: SDL2, SDL2_ttf, libpng
 *   Arch:   sudo pacman -S sdl2 sdl2_ttf ttf-dejavu
 *   Debian: sudo apt install libsdl2-dev libsdl2-ttf-dev fonts-dejavu
 */

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "src/cnn/cnn.h"
#include "src/cnn/model.h"
#include "src/preprocess/image.h"
#include "src/segment/segment.h"
#include "src/solver/solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------
 * Layout constants
 * ---------------------------------------------------------------------- */

#define WIN_W            1280
#define WIN_H             800
#define PANEL_H           115   /**< Height of the top control panel.       */
#define FONT_SIZE          16
#define FONT_SIZE_SM       13

#define BTN_H              36
#define BTN_W             150

/** Row 1 positions */
#define BTN_IMG_X           8
#define BTN_IMG_Y           8
#define PATH_LABEL_W      300
#define BTN_MOD_X         (BTN_IMG_X + BTN_W + 8 + PATH_LABEL_W + 16)
#define BTN_MOD_Y           8

/** Row 2 positions */
#define INPUT_X             8
#define INPUT_Y            52
#define WORDS_INPUT_W     (WIN_W - INPUT_X - BTN_W - 8 - 8)
#define BTN_SEARCH_X      (INPUT_X + WORDS_INPUT_W + 8)
#define BTN_SEARCH_Y       52

#define MAX_RESULTS        64
#define DEFAULT_MODEL_DIR "models/"

/* -------------------------------------------------------------------------
 * Application state
 * ---------------------------------------------------------------------- */

typedef struct {
    SDL_Window   *window;
    SDL_Renderer *renderer;
    TTF_Font     *font;       /**< Normal font (FONT_SIZE).   */
    TTF_Font     *font_sm;    /**< Small font (FONT_SIZE_SM). */

    /* Files */
    char  image_path[512];
    char  model_path[512];

    /* Original image — NEVER modified after load */
    Image        *orig_img;
    SDL_Texture  *orig_tex;

    /* CNN model */
    CNN          *net;

    /* Words text input */
    char  words_buf[2048];
    int   words_focused;

    /* OCR results */
    BoundingBox  *cells;     /**< Sorted bounding boxes (image coords). */
    size_t        n_cells;
    int           grid_rows;
    int           grid_cols;
    int           cell_pitch; /**< Estimated grid pitch in image pixels. */

    /* Word-search results */
    char       result_words[MAX_RESULTS][64];
    WordResult results[MAX_RESULTS];
    int        n_results;

    /* Image display geometry (updated on load / window resize) */
    int   disp_x, disp_y;  /**< Top-left corner of image in window.     */
    float disp_scale;       /**< image_pixel * scale = screen_pixel.     */

    /* UI state */
    char  status[256];
    int   status_ok;     /**< 1=green, 0=red, -1=neutral grey.           */
    int   hovered_btn;   /**< 0=none, 1=open_img, 2=open_mod, 3=search. */
    int   running;
    int   busy;          /**< Set during OCR to show "…" on button.      */
} GuiState;

/* -------------------------------------------------------------------------
 * Font search
 * ---------------------------------------------------------------------- */

static const char *const FONT_PATHS[] = {
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/FreeSans.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.ttf",
    "/usr/share/fonts/freefont/FreeSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    NULL
};

static TTF_Font *find_font(int size)
{
    struct stat st;
    for (int i = 0; FONT_PATHS[i]; i++) {
        if (stat(FONT_PATHS[i], &st) == 0) {
            TTF_Font *f = TTF_OpenFont(FONT_PATHS[i], size);
            if (f) return f;
        }
    }
    return NULL;
}

/* -------------------------------------------------------------------------
 * File dialog (tries zenity, kdialog, yad in order)
 * ---------------------------------------------------------------------- */

static int open_file_dialog(char *out, size_t out_len)
{
    static const char *const CMDS[] = {
        "zenity --file-selection 2>/dev/null",
        "kdialog --getopenfilename . 2>/dev/null",
        "yad --file-selection 2>/dev/null",
        NULL
    };

    for (int i = 0; CMDS[i]; i++) {
        FILE *fp = popen(CMDS[i], "r");
        if (!fp) continue;
        char buf[512] = {0};
        int got = (fgets(buf, sizeof(buf), fp) != NULL);
        pclose(fp);
        if (!got) continue;
        /* Strip trailing newline */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r'))
            buf[--len] = '\0';
        if (len > 0) {
            strncpy(out, buf, out_len - 1);
            out[out_len - 1] = '\0';
            return 0;
        }
    }
    return -1;
}

/* -------------------------------------------------------------------------
 * Drawing helpers
 * ---------------------------------------------------------------------- */

static void fill_rect(SDL_Renderer *r, int x, int y, int w, int h,
                      Uint8 cr, Uint8 cg, Uint8 cb, Uint8 ca)
{
    SDL_SetRenderDrawColor(r, cr, cg, cb, ca);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(r, &rect);
}

static void outline_rect(SDL_Renderer *r, int x, int y, int w, int h,
                         Uint8 cr, Uint8 cg, Uint8 cb, Uint8 ca, int t)
{
    SDL_SetRenderDrawColor(r, cr, cg, cb, ca);
    for (int k = 0; k < t; k++) {
        SDL_Rect rect = {x+k, y+k, w-2*k, h-2*k};
        SDL_RenderDrawRect(r, &rect);
    }
}

static void draw_text_at(GuiState *g, TTF_Font *f, const char *txt,
                         int x, int y, Uint8 cr, Uint8 cg, Uint8 cb)
{
    if (!f || !txt || !*txt) return;
    SDL_Color c = {cr, cg, cb, 255};
    SDL_Surface *s = TTF_RenderUTF8_Blended(f, txt, c);
    if (!s) return;
    SDL_Texture *t = SDL_CreateTextureFromSurface(g->renderer, s);
    SDL_FreeSurface(s);
    if (!t) return;
    int tw, th;
    SDL_QueryTexture(t, NULL, NULL, &tw, &th);
    SDL_Rect dst = {x, y, tw, th};
    SDL_RenderCopy(g->renderer, t, NULL, &dst);
    SDL_DestroyTexture(t);
}

/** Draw text horizontally centred inside a BTN_H-tall button at (bx, by, bw). */
static void draw_btn_label(GuiState *g, const char *txt,
                           int bx, int by, int bw)
{
    if (!g->font || !txt || !*txt) return;
    SDL_Color c = {230, 230, 230, 255};
    SDL_Surface *s = TTF_RenderUTF8_Blended(g->font, txt, c);
    if (!s) return;
    SDL_Texture *t = SDL_CreateTextureFromSurface(g->renderer, s);
    SDL_FreeSurface(s);
    if (!t) return;
    int tw, th;
    SDL_QueryTexture(t, NULL, NULL, &tw, &th);
    SDL_Rect dst = {bx + (bw - tw) / 2, by + (BTN_H - th) / 2, tw, th};
    SDL_RenderCopy(g->renderer, t, NULL, &dst);
    SDL_DestroyTexture(t);
}

/* -------------------------------------------------------------------------
 * Display geometry
 * ---------------------------------------------------------------------- */

static void update_geometry(GuiState *g)
{
    if (!g->orig_img) return;
    int aw = WIN_W;
    int ah = WIN_H - PANEL_H;
    float sx = (float)aw / g->orig_img->width;
    float sy = (float)ah / g->orig_img->height;
    g->disp_scale = sx < sy ? sx : sy;
    int dw = (int)(g->orig_img->width  * g->disp_scale);
    int dh = (int)(g->orig_img->height * g->disp_scale);
    g->disp_x = (aw - dw) / 2;
    g->disp_y = PANEL_H + (ah - dh) / 2;
}

static int to_screen_x(GuiState *g, int ix)
{
    return g->disp_x + (int)(ix * g->disp_scale);
}
static int to_screen_y(GuiState *g, int iy)
{
    return g->disp_y + (int)(iy * g->disp_scale);
}

/* -------------------------------------------------------------------------
 * Image / model loading
 * ---------------------------------------------------------------------- */

static int gui_load_image(GuiState *g, const char *path)
{
    if (g->orig_tex) { SDL_DestroyTexture(g->orig_tex); g->orig_tex = NULL; }
    if (g->orig_img) { image_free(g->orig_img); g->orig_img = NULL; }
    if (g->cells)    { free(g->cells); g->cells = NULL; g->n_cells = 0; }
    g->n_results = 0;

    g->orig_img = image_load_png(path);
    if (!g->orig_img) {
        snprintf(g->status, sizeof(g->status),
                 "Erreur: impossible de charger '%s'", path);
        g->status_ok = 0;
        return -1;
    }

    /* Build SDL texture from the original RGBA pixels (no copy needed). */
    SDL_Surface *surf = SDL_CreateRGBSurfaceWithFormatFrom(
        g->orig_img->pixels,
        g->orig_img->width, g->orig_img->height,
        32, g->orig_img->width * 4,
        SDL_PIXELFORMAT_RGBA32
    );
    if (surf) {
        g->orig_tex = SDL_CreateTextureFromSurface(g->renderer, surf);
        SDL_FreeSurface(surf);
    }
    if (!g->orig_tex) {
        snprintf(g->status, sizeof(g->status),
                 "Erreur texture SDL: %s", SDL_GetError());
        g->status_ok = 0;
        return -1;
    }

    strncpy(g->image_path, path, sizeof(g->image_path) - 1);
    update_geometry(g);
    snprintf(g->status, sizeof(g->status), "Image: %dx%d px",
             g->orig_img->width, g->orig_img->height);
    g->status_ok = 1;
    return 0;
}

static int gui_load_model(GuiState *g, const char *path)
{
    if (!g->net) {
        g->net = cnn_create();
        if (!g->net) {
            snprintf(g->status, sizeof(g->status),
                     "Erreur: allocation CNN échouée");
            g->status_ok = 0;
            return -1;
        }
    }
    if (model_load(g->net, path) != 0) {
        snprintf(g->status, sizeof(g->status),
                 "Erreur: chargement modèle '%s'", path);
        g->status_ok = 0;
        return -1;
    }
    strncpy(g->model_path, path, sizeof(g->model_path) - 1);
    const char *name = strrchr(path, '/');
    snprintf(g->status, sizeof(g->status),
             "Modèle: %s", name ? name + 1 : path);
    g->status_ok = 1;
    return 0;
}

/* -------------------------------------------------------------------------
 * OCR pipeline  (mirrors solve_main.c logic)
 * ---------------------------------------------------------------------- */

#define TTA_N  5

static void forward_region(const Image *gray,
                            int x1, int y1, int x2, int y2,
                            CNN *net, float *probs)
{
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > gray->width)  x2 = gray->width;
    if (y2 > gray->height) y2 = gray->height;
    int cw = x2 - x1, ch = y2 - y1;
    if (cw <= 0 || ch <= 0) return;

    Image cell;
    cell.width  = cw;
    cell.height = ch;
    cell.pixels = malloc((size_t)cw * ch * 4);
    if (!cell.pixels) return;

    for (int row = 0; row < ch; row++) {
        const unsigned char *src =
            gray->pixels + ((y1 + row) * gray->width + x1) * 4;
        memcpy(cell.pixels + row * cw * 4, src, (size_t)cw * 4);
    }

    image_binarize(&cell);
    Image *rsz = image_resize(&cell, CNN_IMG_W, CNN_IMG_H);
    free(cell.pixels);
    if (!rsz) return;

    float px[CNN_IMG_H * CNN_IMG_W];
    image_to_float(rsz, px);
    image_free(rsz);

    cnn_forward(net, px);
    for (int k = 0; k < CNN_N_CLASSES; k++)
        probs[k] += net->act.output[k];
}

static int recognise_cell(const Image *gray, const BoundingBox *box,
                           int cell_size, CNN *net)
{
    int cx   = box->x + box->w / 2;
    int cy   = box->y + box->h / 2;
    int half = (cell_size > 0)
        ? cell_size / 2
        : (int)((box->w > box->h ? box->w : box->h) * 0.35f) + box->w / 2;
    if (half < 4) half = 4;

    static const int shifts[TTA_N][2] = {{0,0},{-2,0},{2,0},{0,-2},{0,2}};
    float probs[CNN_N_CLASSES] = {0};
    for (int t = 0; t < TTA_N; t++) {
        int dx = shifts[t][0], dy = shifts[t][1];
        forward_region(gray,
                       cx - half + dx, cy - half + dy,
                       cx + half + dx, cy + half + dy,
                       net, probs);
    }
    int best = 0;
    for (int k = 1; k < CNN_N_CLASSES; k++)
        if (probs[k] > probs[best]) best = k;
    return best;
}

static void gui_run_ocr(GuiState *g)
{
    if (!g->orig_img || !g->net) {
        snprintf(g->status, sizeof(g->status),
                 "Chargez une image et un modèle d'abord.");
        g->status_ok = 0;
        return;
    }

    /* Clear previous results */
    if (g->cells) { free(g->cells); g->cells = NULL; g->n_cells = 0; }
    g->n_results = 0;

    /* Load a fresh copy of the image for grayscale processing. */
    Image *gray = image_load_png(g->image_path);
    if (!gray) {
        snprintf(g->status, sizeof(g->status), "Erreur: rechargement image");
        g->status_ok = 0;
        return;
    }
    image_to_grayscale(gray);

    /* Build binarized single-channel buffer for the segmenter. */
    int npx = gray->width * gray->height;
    unsigned char *bin = malloc((size_t)npx);
    if (!bin) { image_free(gray); return; }
    {
        long sum = 0;
        for (int i = 0; i < npx; i++) sum += gray->pixels[i * 4];
        unsigned char thr = (unsigned char)(sum / npx);
        for (int i = 0; i < npx; i++)
            bin[i] = (gray->pixels[i * 4] > thr) ? 255 : 0;
    }

    snprintf(g->status, sizeof(g->status), "Segmentation en cours…");

    SegmentResult *seg = segment_image(bin, gray->width, gray->height);
    free(bin);

    if (!seg || seg->count == 0) {
        image_free(gray);
        if (seg) segment_result_free(seg);
        snprintf(g->status, sizeof(g->status),
                 "Erreur: aucune cellule détectée");
        g->status_ok = 0;
        return;
    }

    /* Estimate grid cell size (pitch). */
    int cell_size = 0, row_count = 1, col_count = 1;
    if (seg->count > 1) {
        int tol = seg->cells[0].h > 0 ? seg->cells[0].h * 3 / 5 : 20;
        int prev = seg->cells[0].y + seg->cells[0].h / 2;
        for (size_t i = 1; i < seg->count; i++) {
            int cy = seg->cells[i].y + seg->cells[i].h / 2;
            if (cy - prev > tol) { row_count++; prev = cy; }
        }
        col_count = row_count > 0 ? (int)seg->count / row_count : 1;

        if (row_count > 1 && col_count > 1 &&
            (size_t)(row_count * col_count) == seg->count) {
            int cy0 = seg->cells[0].y + seg->cells[0].h / 2;
            int cyN = seg->cells[(row_count-1)*col_count].y
                    + seg->cells[(row_count-1)*col_count].h / 2;
            int cx0 = seg->cells[0].x + seg->cells[0].w / 2;
            int cxN = seg->cells[col_count-1].x + seg->cells[col_count-1].w / 2;
            cell_size = ((cyN - cy0) / (row_count - 1)
                       + (cxN - cx0) / (col_count - 1)) / 2;
        }
    }
    g->cell_pitch = cell_size;

    /* Recognise each cell. */
    snprintf(g->status, sizeof(g->status),
             "Reconnaissance (%zu cellules)…", seg->count);

    int *labels = malloc(seg->count * sizeof(int));
    if (!labels) {
        segment_result_free(seg); image_free(gray); return;
    }
    for (size_t i = 0; i < seg->count; i++)
        labels[i] = recognise_cell(gray, &seg->cells[i], cell_size, g->net);

    image_free(gray);

    /* Infer grid dimensions. */
    int rows, cols;
    if (seg->rows > 0 && seg->cols > 0) {
        rows = seg->rows; cols = seg->cols;
    } else {
        rows = 1;
        if (seg->count > 1) {
            int tol = seg->cells[0].h > 0 ? seg->cells[0].h * 3 / 5 : 20;
            int prev = seg->cells[0].y + seg->cells[0].h / 2;
            for (size_t i = 1; i < seg->count; i++) {
                int cy = seg->cells[i].y + seg->cells[i].h / 2;
                if (cy - prev > tol) { rows++; prev = cy; }
            }
        }
        cols = rows > 0 ? (int)seg->count / rows : (int)seg->count;
    }
    g->grid_rows = rows;
    g->grid_cols = cols;

    /* Keep a copy of sorted bounding boxes for highlight rendering. */
    g->cells   = malloc(seg->count * sizeof(BoundingBox));
    g->n_cells = seg->count;
    if (g->cells)
        memcpy(g->cells, seg->cells, seg->count * sizeof(BoundingBox));

    /* Build character grid. */
    CharGrid *grid = grid_create(rows, cols);
    if (grid) grid_fill(grid, labels, seg->count);
    free(labels);
    segment_result_free(seg);

    if (!grid) {
        snprintf(g->status, sizeof(g->status),
                 "Erreur: création de la grille");
        g->status_ok = 0;
        return;
    }

    /* Print grid to stdout (helpful for debugging). */
    printf("\nGrille reconnue (%d×%d):\n", rows, cols);
    grid_print(grid);

    /* Word search. */
    g->n_results = 0;
    if (g->words_buf[0]) {
        char tmp[2048];
        strncpy(tmp, g->words_buf, sizeof(tmp) - 1);
        char *tok = strtok(tmp, ",");
        while (tok && g->n_results < MAX_RESULTS) {
            while (*tok == ' ') tok++;
            char *end = tok + strlen(tok) - 1;
            while (end > tok && *end == ' ') *end-- = '\0';
            if (*tok) {
                strncpy(g->result_words[g->n_results], tok,
                        sizeof(g->result_words[0]) - 1);
                g->results[g->n_results] = solver_find(grid, tok);
                g->n_results++;
            }
            tok = strtok(NULL, ",");
        }
    }

    int found = 0;
    for (int i = 0; i < g->n_results; i++) {
        if (g->results[i].found) found++;
        printf("  %-20s %s\n",
               g->result_words[i],
               g->results[i].found ? "trouvé" : "non trouvé");
    }

    grid_free(grid);

    if (g->n_results > 0) {
        snprintf(g->status, sizeof(g->status),
                 "Grille %d×%d — %d/%d mot(s) trouvé(s)",
                 rows, cols, found, g->n_results);
        g->status_ok = (found == g->n_results) ? 1 : -1;
    } else {
        snprintf(g->status, sizeof(g->status),
                 "Grille %d×%d reconnue — entrez des mots à chercher",
                 rows, cols);
        g->status_ok = 1;
    }
}

/* -------------------------------------------------------------------------
 * Word highlight rendering
 * ---------------------------------------------------------------------- */

static void draw_word_highlight(GuiState *g, const WordResult *r)
{
    if (!r->found || !g->cells || g->grid_cols <= 0) return;

    int dr = (r->end_r > r->start_r) ? 1 : (r->end_r < r->start_r) ? -1 : 0;
    int dc = (r->end_c > r->start_c) ? 1 : (r->end_c < r->start_c) ? -1 : 0;

    int row = r->start_r, col = r->start_c;
    for (;;) {
        int idx = row * g->grid_cols + col;
        if (idx >= 0 && (size_t)idx < g->n_cells) {
            BoundingBox *b = &g->cells[idx];
            int cx_img = b->x + b->w / 2;
            int cy_img = b->y + b->h / 2;
            int half   = g->cell_pitch > 0
                ? g->cell_pitch / 2
                : (b->w > b->h ? b->w : b->h) / 2 + 4;

            int sx = to_screen_x(g, cx_img - half);
            int sy = to_screen_y(g, cy_img - half);
            int sw = (int)((half * 2) * g->disp_scale);
            int sh = (int)((half * 2) * g->disp_scale);
            if (sw < 4) sw = 4;
            if (sh < 4) sh = 4;

            SDL_SetRenderDrawBlendMode(g->renderer, SDL_BLENDMODE_BLEND);
            fill_rect(g->renderer, sx, sy, sw, sh, 220, 50, 50, 120);
            outline_rect(g->renderer, sx, sy, sw, sh, 255, 30, 30, 255, 2);
        }

        if (row == r->end_r && col == r->end_c) break;
        row += dr;
        col += dc;
    }
}

/* -------------------------------------------------------------------------
 * Render frame
 * ---------------------------------------------------------------------- */

static void gui_render(GuiState *g)
{
    /* ---- Background ---- */
    SDL_SetRenderDrawColor(g->renderer, 38, 38, 42, 255);
    SDL_RenderClear(g->renderer);

    /* ---- Panel ---- */
    fill_rect(g->renderer, 0, 0, WIN_W, PANEL_H, 52, 52, 58, 255);

    /* Separator line */
    SDL_SetRenderDrawColor(g->renderer, 70, 70, 80, 255);
    SDL_RenderDrawLine(g->renderer, 0, PANEL_H - 1, WIN_W, PANEL_H - 1);

    /* ---- Row 1: Open Image ---- */
    {
        Uint8 r = (g->hovered_btn == 1) ? 100 : 72;
        Uint8 gr = (g->hovered_btn == 1) ? 140 : 110;
        Uint8 b = (g->hovered_btn == 1) ? 220 : 195;
        fill_rect(g->renderer, BTN_IMG_X, BTN_IMG_Y, BTN_W, BTN_H, r, gr, b, 255);
        draw_btn_label(g, "Ouvrir image", BTN_IMG_X, BTN_IMG_Y, BTN_W);
    }

    /* Image path label box */
    {
        int px = BTN_IMG_X + BTN_W + 8, py = BTN_IMG_Y;
        fill_rect(g->renderer, px, py, PATH_LABEL_W, BTN_H, 28, 28, 32, 255);
        if (g->font_sm) {
            const char *lbl = g->image_path[0]
                ? (strrchr(g->image_path, '/') ? strrchr(g->image_path, '/') + 1
                                               : g->image_path)
                : "(aucune image)";
            draw_text_at(g, g->font_sm, lbl,
                         px + 6, py + (BTN_H - FONT_SIZE_SM) / 2,
                         160, 160, 160);
        }
    }

    /* ---- Row 1: Open Model ---- */
    {
        Uint8 r = (g->hovered_btn == 2) ? 100 : 72;
        Uint8 gr = (g->hovered_btn == 2) ? 140 : 110;
        Uint8 b = (g->hovered_btn == 2) ? 220 : 195;
        fill_rect(g->renderer, BTN_MOD_X, BTN_MOD_Y, BTN_W, BTN_H, r, gr, b, 255);
        draw_btn_label(g, "Ouvrir modèle", BTN_MOD_X, BTN_MOD_Y, BTN_W);
    }

    /* Model path label box */
    {
        int px = BTN_MOD_X + BTN_W + 8, py = BTN_MOD_Y;
        int pw = WIN_W - px - 8;
        if (pw > 0) {
            fill_rect(g->renderer, px, py, pw, BTN_H, 28, 28, 32, 255);
            if (g->font_sm) {
                const char *lbl = g->model_path[0]
                    ? (strrchr(g->model_path, '/') ? strrchr(g->model_path, '/') + 1
                                                   : g->model_path)
                    : "(aucun modèle — auto-détecté)";
                draw_text_at(g, g->font_sm, lbl,
                             px + 6, py + (BTN_H - FONT_SIZE_SM) / 2,
                             160, 160, 160);
            }
        }
    }

    /* ---- Row 2: Words input ---- */
    {
        Uint8 ibr = g->words_focused ? 42 : 28;
        fill_rect(g->renderer, INPUT_X, INPUT_Y, WORDS_INPUT_W, BTN_H,
                  ibr, ibr, ibr + 4, 255);
        outline_rect(g->renderer, INPUT_X, INPUT_Y, WORDS_INPUT_W, BTN_H,
                     g->words_focused ? 90 : 60,
                     g->words_focused ? 130 : 60,
                     g->words_focused ? 210 : 70, 255, 1);

        if (g->font) {
            if (g->words_buf[0]) {
                draw_text_at(g, g->font, g->words_buf,
                             INPUT_X + 8, INPUT_Y + (BTN_H - FONT_SIZE) / 2,
                             220, 220, 220);
            } else {
                draw_text_at(g, g->font,
                             "Mots à chercher (séparés par des virgules)…",
                             INPUT_X + 8, INPUT_Y + (BTN_H - FONT_SIZE) / 2,
                             90, 90, 100);
            }

            /* Text cursor blink */
            if (g->words_focused && (SDL_GetTicks() / 530) % 2 == 0) {
                int tw = 0;
                TTF_SizeUTF8(g->font, g->words_buf, &tw, NULL);
                int cx = INPUT_X + 8 + tw + 1;
                int cy = INPUT_Y + 6;
                SDL_SetRenderDrawColor(g->renderer, 200, 200, 200, 255);
                SDL_RenderDrawLine(g->renderer, cx, cy, cx, cy + FONT_SIZE + 2);
            }
        }
    }

    /* ---- Row 2: Search button ---- */
    {
        Uint8 r = g->busy ? 45 : (g->hovered_btn == 3) ? 100 : 72;
        Uint8 gr = g->busy ? 45 : (g->hovered_btn == 3) ? 140 : 110;
        Uint8 b = g->busy ? 50 : (g->hovered_btn == 3) ? 220 : 195;
        fill_rect(g->renderer, BTN_SEARCH_X, BTN_SEARCH_Y, BTN_W, BTN_H,
                  r, gr, b, 255);
        draw_btn_label(g, g->busy ? "…" : "Chercher",
                       BTN_SEARCH_X, BTN_SEARCH_Y, BTN_W);
    }

    /* ---- Status bar ---- */
    if (g->font_sm && g->status[0]) {
        Uint8 sr = (g->status_ok == 1)  ? 80  :
                   (g->status_ok == 0)  ? 220 : 170;
        Uint8 sgg = (g->status_ok == 1) ? 200 :
                    (g->status_ok == 0) ?  80 : 170;
        Uint8 sb = (g->status_ok == 1)  ?  80 :
                   (g->status_ok == 0)  ?  80 : 180;
        draw_text_at(g, g->font_sm, g->status,
                     8, PANEL_H - FONT_SIZE_SM - 6, sr, sgg, sb);
    }

    /* ---- Image area ---- */
    if (g->orig_tex) {
        int dw = (int)(g->orig_img->width  * g->disp_scale);
        int dh = (int)(g->orig_img->height * g->disp_scale);
        SDL_Rect dst = {g->disp_x, g->disp_y, dw, dh};
        SDL_RenderCopy(g->renderer, g->orig_tex, NULL, &dst);

        /* Word highlights */
        for (int i = 0; i < g->n_results; i++)
            draw_word_highlight(g, &g->results[i]);
    } else {
        fill_rect(g->renderer, 0, PANEL_H, WIN_W, WIN_H - PANEL_H, 42, 42, 46, 255);
        if (g->font) {
            draw_text_at(g, g->font,
                         "Ouvrez une image pour commencer",
                         WIN_W / 2 - 150,
                         PANEL_H + (WIN_H - PANEL_H) / 2 - FONT_SIZE / 2,
                         80, 80, 90);
        }
    }

    SDL_RenderPresent(g->renderer);
}

/* -------------------------------------------------------------------------
 * Event handling
 * ---------------------------------------------------------------------- */

static int is_btn_hit(int mx, int my, int bx, int by)
{
    return mx >= bx && mx < bx + BTN_W &&
           my >= by && my < by + BTN_H;
}

static void handle_motion(GuiState *g, int mx, int my)
{
    g->hovered_btn = 0;
    if      (is_btn_hit(mx, my, BTN_IMG_X,    BTN_IMG_Y))    g->hovered_btn = 1;
    else if (is_btn_hit(mx, my, BTN_MOD_X,    BTN_MOD_Y))    g->hovered_btn = 2;
    else if (is_btn_hit(mx, my, BTN_SEARCH_X, BTN_SEARCH_Y)) g->hovered_btn = 3;
}

static void handle_click(GuiState *g, int mx, int my)
{
    /* Words-input focus */
    int was_focused = g->words_focused;
    g->words_focused = (mx >= INPUT_X && mx < INPUT_X + WORDS_INPUT_W &&
                        my >= INPUT_Y && my < INPUT_Y + BTN_H);
    if (g->words_focused && !was_focused)
        SDL_StartTextInput();
    else if (!g->words_focused && was_focused)
        SDL_StopTextInput();

    if (is_btn_hit(mx, my, BTN_IMG_X, BTN_IMG_Y)) {
        char path[512] = {0};
        if (open_file_dialog(path, sizeof(path)) == 0 && path[0]) {
            gui_load_image(g, path);
        } else {
            snprintf(g->status, sizeof(g->status),
                     "Pas d'outil de dialogue (installez zenity ou kdialog)");
            g->status_ok = 0;
        }

    } else if (is_btn_hit(mx, my, BTN_MOD_X, BTN_MOD_Y)) {
        char path[512] = {0};
        if (open_file_dialog(path, sizeof(path)) == 0 && path[0]) {
            gui_load_model(g, path);
        } else {
            snprintf(g->status, sizeof(g->status),
                     "Pas d'outil de dialogue (installez zenity ou kdialog)");
            g->status_ok = 0;
        }

    } else if (is_btn_hit(mx, my, BTN_SEARCH_X, BTN_SEARCH_Y) && !g->busy) {
        g->busy = 1;
        gui_render(g);   /* show "…" immediately */
        gui_run_ocr(g);
        g->busy = 0;
    }
}

static void handle_keydown(GuiState *g, SDL_Keycode key)
{
    if (!g->words_focused) return;

    if (key == SDLK_BACKSPACE) {
        size_t len = strlen(g->words_buf);
        if (len > 0) g->words_buf[len - 1] = '\0';

    } else if (key == SDLK_RETURN || key == SDLK_KP_ENTER) {
        g->words_focused = 0;
        SDL_StopTextInput();
        if (!g->busy) {
            g->busy = 1;
            gui_render(g);
            gui_run_ocr(g);
            g->busy = 0;
        }

    } else if (key == SDLK_ESCAPE) {
        g->words_focused = 0;
        SDL_StopTextInput();
    }
}

static void handle_text_input(GuiState *g, const char *text)
{
    if (!g->words_focused) return;
    size_t cur = strlen(g->words_buf);
    size_t add = strlen(text);
    if (cur + add < sizeof(g->words_buf) - 1)
        strncat(g->words_buf, text, sizeof(g->words_buf) - cur - 1);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    const char *cli_model = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            cli_model = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: ./gui [--model <path>]\n"
                   "  --model <path>  Use this model file instead of auto-detecting.\n");
            return 0;
        }
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    if (TTF_Init() != 0) {
        fprintf(stderr, "TTF_Init: %s\n", TTF_GetError());
        SDL_Quit();
        return 1;
    }

    GuiState g;
    memset(&g, 0, sizeof(g));
    g.running   = 1;
    g.status_ok = -1;
    snprintf(g.status, sizeof(g.status), "Prêt");

    g.window = SDL_CreateWindow(
        "OCR — Solveur de mots mêlés",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H,
        SDL_WINDOW_SHOWN
    );
    if (!g.window) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit(); SDL_Quit();
        return 1;
    }

    g.renderer = SDL_CreateRenderer(
        g.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!g.renderer) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(g.window);
        TTF_Quit(); SDL_Quit();
        return 1;
    }
    SDL_SetRenderDrawBlendMode(g.renderer, SDL_BLENDMODE_BLEND);

    g.font    = find_font(FONT_SIZE);
    g.font_sm = find_font(FONT_SIZE_SM);
    if (!g.font)
        fprintf(stderr,
                "Warning: no TTF font found — text will not render.\n"
                "  Arch:   sudo pacman -S ttf-dejavu\n"
                "  Debian: sudo apt install fonts-dejavu\n");

    /* Auto-load latest model. */
    g.net = cnn_create();
    if (g.net) {
        char mpath[512] = {0};
        if (cli_model) {
            strncpy(mpath, cli_model, sizeof(mpath) - 1);
        } else {
            model_find_latest(DEFAULT_MODEL_DIR, mpath, sizeof(mpath));
        }
        if (mpath[0] && model_load(g.net, mpath) == 0) {
            strncpy(g.model_path, mpath, sizeof(g.model_path) - 1);
            const char *name = strrchr(mpath, '/');
            snprintf(g.status, sizeof(g.status),
                     "Modèle auto: %s", name ? name + 1 : mpath);
            g.status_ok = 1;
        }
    }

    /* Event loop. */
    SDL_Event ev;
    while (g.running) {
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                g.running = 0;
                break;
            case SDL_MOUSEMOTION:
                handle_motion(&g, ev.motion.x, ev.motion.y);
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (ev.button.button == SDL_BUTTON_LEFT)
                    handle_click(&g, ev.button.x, ev.button.y);
                break;
            case SDL_KEYDOWN:
                if (ev.key.keysym.sym == SDLK_q &&
                    (ev.key.keysym.mod & KMOD_CTRL))
                    g.running = 0;
                else
                    handle_keydown(&g, ev.key.keysym.sym);
                break;
            case SDL_TEXTINPUT:
                handle_text_input(&g, ev.text.text);
                break;
            }
        }

        gui_render(&g);
        SDL_Delay(16);
    }

    /* Cleanup. */
    if (g.font)    TTF_CloseFont(g.font);
    if (g.font_sm) TTF_CloseFont(g.font_sm);
    if (g.orig_tex) SDL_DestroyTexture(g.orig_tex);
    if (g.orig_img) image_free(g.orig_img);
    if (g.net)      cnn_free(g.net);
    if (g.cells)    free(g.cells);

    SDL_DestroyRenderer(g.renderer);
    SDL_DestroyWindow(g.window);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
