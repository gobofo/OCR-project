/**
 * @file gui_main.c
 * @brief SDL2 graphical interface for the OCR crossword solver.
 *
 * Provides a 1280×800 window with three text-input rows:
 *   - **Image**  — path to the PNG crossword image
 *   - **Modèle** — path to the trained CNN model (.bin)
 *   - **Mots**   — comma-separated list of words to find
 *
 * Clicking **Charger** (or pressing Enter) loads the file.
 * Clicking **Chercher** (or pressing Enter in the words field) runs the full
 * OCR pipeline and overlays red rectangles on each found-word cell.
 * The original image pixels are never modified.
 *
 * @par Keyboard shortcuts
 *   - **Tab**    — cycle focus between fields
 *   - **Ctrl+V** — paste from clipboard into the focused field
 *   - **Enter**  — validate / trigger action for the focused field
 *   - **Escape** — clear focus
 *   - **Ctrl+Q** — quit
 *
 * @par Dependencies
 *   SDL2, SDL2_ttf, libpng — plus the project's own CNN / segment / solver.
 *
 * @par Usage
 * @code
 *   ./gui                          # auto-detects latest model in models/
 *   ./gui --model models/foo.bin   # explicit model path
 * @endcode
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

/** @defgroup layout Layout constants
 *  Pixel dimensions that define the window and control-panel geometry.
 *  @{
 */
#define WIN_W          1280  /**< Window width in pixels.                       */
#define WIN_H           800  /**< Window height in pixels.                      */
#define ROW_H            36  /**< Height of one input row (label + field + btn).*/
#define ROW_PAD           8  /**< Vertical gap between consecutive rows.        */
#define LABEL_W          68  /**< Pixel width reserved for the row label.       */
#define BTN_W           120  /**< Width of each action button.                  */
#define FONT_SIZE        15  /**< Point size for the primary font.              */
#define FONT_SIZE_SM     13  /**< Point size for the smaller (label/status) font.*/

/** Top-left Y of the Image row. */
#define ROW1_Y           8
/** Top-left Y of the Modèle row. */
#define ROW2_Y          (ROW1_Y + ROW_H + ROW_PAD)
/** Top-left Y of the Mots row. */
#define ROW3_Y          (ROW2_Y + ROW_H + ROW_PAD)
/** Y of the status-bar text. */
#define STATUS_Y        (ROW3_Y + ROW_H + 6)
/** Total height of the top control panel. */
#define PANEL_H         (STATUS_Y + FONT_SIZE_SM + 8)

/** X origin of all text-input fields (after the label). */
#define INPUT_X         (LABEL_W + 4)
/** Width of all text-input fields. */
#define INPUT_W         (WIN_W - INPUT_X - BTN_W - 8 - 8)
/** X origin of all action buttons. */
#define BTN_X           (INPUT_X + INPUT_W + 8)

/** Maximum number of word-search results kept simultaneously. */
#define MAX_RESULTS      64
/** Default directory scanned for the most recent model file. */
#define DEFAULT_MODEL_DIR "models/"
/** @} */

/** @defgroup focus Focus constants
 *  Identify which text field currently holds keyboard focus.
 *  @{
 */
#define FOCUS_NONE   0  /**< No field focused.       */
#define FOCUS_IMAGE  1  /**< Image-path field.        */
#define FOCUS_MODEL  2  /**< Model-path field.        */
#define FOCUS_WORDS  3  /**< Word-list field.         */
/** @} */

/* -------------------------------------------------------------------------
 * State
 * ---------------------------------------------------------------------- */

/**
 * @brief Complete application state passed to every GUI function.
 *
 * All mutable state lives here so that every helper function receives only
 * a single pointer instead of a growing parameter list.
 */
typedef struct {
    SDL_Window   *window;    /**< Main application window.                  */
    SDL_Renderer *renderer;  /**< Hardware-accelerated renderer.            */
    TTF_Font     *font;      /**< Primary font (FONT_SIZE pt).              */
    TTF_Font     *font_sm;   /**< Smaller font for labels/status (FONT_SIZE_SM pt). */

    /* ---- Text input buffers ---- */
    char  image_buf[512];    /**< Editable image-path field content.        */
    char  model_buf[512];    /**< Editable model-path field content.        */
    char  words_buf[2048];   /**< Comma-separated words to search for.      */
    int   focused;           /**< Which field has keyboard focus (FOCUS_*). */

    /* ---- Loaded assets ---- */
    Image        *orig_img;  /**< Original RGBA image — @b never modified.  */
    SDL_Texture  *orig_tex;  /**< GPU texture built once from orig_img.     */
    CNN          *net;       /**< Loaded CNN model.                         */

    /* ---- OCR / solver results ---- */
    BoundingBox  *cells;     /**< Sorted letter bounding boxes (image coords). */
    size_t        n_cells;   /**< Number of entries in @p cells.            */
    int           grid_rows; /**< Number of grid rows inferred by OCR.      */
    int           grid_cols; /**< Number of grid columns inferred by OCR.   */
    int           cell_pitch;/**< Average grid pitch in image pixels.       */

    char       result_words[MAX_RESULTS][64]; /**< Words that were searched. */
    WordResult results[MAX_RESULTS];          /**< Corresponding solver results. */
    int        n_results;    /**< Number of entries in results[].           */

    /* ---- Display geometry (recomputed on image load) ---- */
    int   disp_x;      /**< X pixel of the image top-left corner on screen. */
    int   disp_y;      /**< Y pixel of the image top-left corner on screen. */
    float disp_scale;  /**< Factor: image_pixel × disp_scale = screen_pixel.*/

    /* ---- UI flags ---- */
    char  status[256]; /**< Status-bar message.                             */
    int   status_ok;   /**< Colour hint: 1=green, 0=red, -1=neutral grey.  */
    int   hovered_btn; /**< Button under the mouse: 0=none, 1–3=row index. */
    int   running;     /**< Set to 0 to exit the event loop.               */
    int   busy;        /**< Non-zero while OCR pipeline is running.        */
} GuiState;

/* Forward declaration — gui_render is called from gui_run_ocr for progress updates */
static void gui_render(GuiState *g);

/* -------------------------------------------------------------------------
 * Font
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
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    NULL
};

/**
 * @brief Open the first TTF font found in the system font search paths.
 *
 * Iterates over @p FONT_PATHS and returns the first font that can be opened
 * at the requested point size.  Intended to avoid a hard dependency on a
 * specific font package.
 *
 * @param size  Desired point size.
 * @return      Opened TTF_Font, or NULL if no font was found.
 *              Caller must close with TTF_CloseFont().
 */
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
 * Drawing helpers
 * ---------------------------------------------------------------------- */

/**
 * @brief Draw a solid filled rectangle.
 * @param r          SDL renderer.
 * @param x,y        Top-left corner.
 * @param w,h        Dimensions in pixels.
 * @param cr,cg,cb,ca  RGBA colour components.
 */
static void fill_rect(SDL_Renderer *r, int x, int y, int w, int h,
                      Uint8 cr, Uint8 cg, Uint8 cb, Uint8 ca)
{
    SDL_SetRenderDrawColor(r, cr, cg, cb, ca);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(r, &rect);
}

/**
 * @brief Draw a 1-pixel outline rectangle (no fill).
 * @param r          SDL renderer.
 * @param x,y        Top-left corner.
 * @param w,h        Dimensions in pixels.
 * @param cr,cg,cb,ca  RGBA colour components.
 */
static void outline_rect(SDL_Renderer *r, int x, int y, int w, int h,
                         Uint8 cr, Uint8 cg, Uint8 cb, Uint8 ca)
{
    SDL_SetRenderDrawColor(r, cr, cg, cb, ca);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderDrawRect(r, &rect);
}

/**
 * @brief Render a UTF-8 string at pixel position (x, y).
 *
 * Creates a temporary texture from the rendered glyph surface, copies it to
 * the renderer, then destroys it.  No-op if @p f or @p txt is NULL/empty.
 *
 * @param g        Application state (provides renderer).
 * @param f        Font to use.
 * @param txt      UTF-8 string to render.
 * @param x,y      Top-left pixel of the text.
 * @param cr,cg,cb RGB colour.
 */
static void draw_text(GuiState *g, TTF_Font *f, const char *txt,
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

/**
 * @brief Draw an action button with a centred text label.
 *
 * The button colour changes when hovered (@p btn_id matches
 * @p g->hovered_btn) and dims when @p busy_flag is set.
 *
 * @param g         Application state.
 * @param btn_id    Button identity (1=load image, 2=load model, 3=search).
 * @param label     UTF-8 label string displayed on the button.
 * @param x,y       Top-left pixel of the button (width is always BTN_W).
 * @param busy_flag Non-zero while the OCR pipeline is running (dims button).
 */
static void draw_btn(GuiState *g, int btn_id, const char *label,
                     int x, int y, int busy_flag)
{
    int hov = (g->hovered_btn == btn_id);
    Uint8 r = busy_flag ? 45 : (hov ? 100 : 72);
    Uint8 gr = busy_flag ? 45 : (hov ? 140 : 110);
    Uint8 b  = busy_flag ? 50 : (hov ? 220 : 195);
    fill_rect(g->renderer, x, y, BTN_W, ROW_H, r, gr, b, 255);

    if (g->font) {
        SDL_Color c = {230, 230, 230, 255};
        SDL_Surface *s = TTF_RenderUTF8_Blended(g->font, label, c);
        if (s) {
            SDL_Texture *t = SDL_CreateTextureFromSurface(g->renderer, s);
            SDL_FreeSurface(s);
            if (t) {
                int tw, th;
                SDL_QueryTexture(t, NULL, NULL, &tw, &th);
                SDL_Rect dst = {x + (BTN_W - tw) / 2,
                                y + (ROW_H - th) / 2, tw, th};
                SDL_RenderCopy(g->renderer, t, NULL, &dst);
                SDL_DestroyTexture(t);
            }
        }
    }
}

/**
 * @brief Draw a text-input field with optional placeholder and blinking cursor.
 *
 * The field is highlighted with a blue border when it has focus
 * (@p field_id == @p g->focused).  If @p buf is empty the @p placeholder
 * text is rendered in a dim colour.  A blinking cursor is drawn after the
 * last character when the field is focused.
 *
 * @param g            Application state.
 * @param field_id     FOCUS_IMAGE / FOCUS_MODEL / FOCUS_WORDS.
 * @param buf          Current text content of the field.
 * @param placeholder  Hint text displayed when @p buf is empty.
 * @param row_y        Top-left Y of the row (field is placed at INPUT_X).
 */
static void draw_input(GuiState *g, int field_id, const char *buf,
                       const char *placeholder, int row_y)
{
    int focused = (g->focused == field_id);
    Uint8 bg = focused ? 42 : 28;
    fill_rect(g->renderer, INPUT_X, row_y, INPUT_W, ROW_H, bg, bg, bg+4, 255);
    Uint8 bc = focused ? 100 : 55;
    Uint8 bgc = focused ? 140 : 55;
    Uint8 bbc = focused ? 210 : 65;
    outline_rect(g->renderer, INPUT_X, row_y, INPUT_W, ROW_H, bc, bgc, bbc, 255);

    if (g->font) {
        int ty = row_y + (ROW_H - FONT_SIZE) / 2;
        if (buf[0]) {
            draw_text(g, g->font, buf, INPUT_X + 8, ty, 220, 220, 220);
        } else {
            draw_text(g, g->font, placeholder, INPUT_X + 8, ty, 80, 80, 90);
        }
        /* Cursor blink */
        if (focused && (SDL_GetTicks() / 530) % 2 == 0) {
            int tw = 0;
            TTF_SizeUTF8(g->font, buf, &tw, NULL);
            int cx = INPUT_X + 8 + tw + 1;
            SDL_SetRenderDrawColor(g->renderer, 200, 200, 200, 255);
            SDL_RenderDrawLine(g->renderer, cx, row_y + 6,
                               cx, row_y + ROW_H - 6);
        }
    }
}

/* -------------------------------------------------------------------------
 * Display geometry
 * ---------------------------------------------------------------------- */

/**
 * @brief Recompute the image display rectangle after a load or window change.
 *
 * Calculates the uniform scale factor that fits @p g->orig_img inside the
 * image area (below PANEL_H) while preserving the aspect ratio, then stores
 * the top-left offset (@p g->disp_x, @p g->disp_y) and @p g->disp_scale.
 *
 * @param g  Application state; @p g->orig_img must be non-NULL.
 */
static void update_geometry(GuiState *g)
{
    if (!g->orig_img) return;
    int aw = WIN_W, ah = WIN_H - PANEL_H;
    float sx = (float)aw / g->orig_img->width;
    float sy = (float)ah / g->orig_img->height;
    g->disp_scale = sx < sy ? sx : sy;
    int dw = (int)(g->orig_img->width  * g->disp_scale);
    int dh = (int)(g->orig_img->height * g->disp_scale);
    g->disp_x = (aw - dw) / 2;
    g->disp_y = PANEL_H + (ah - dh) / 2;
}

/** @brief Convert an image-space X coordinate to a screen X coordinate. */
static int to_sx(GuiState *g, int ix) { return g->disp_x + (int)(ix * g->disp_scale); }
/** @brief Convert an image-space Y coordinate to a screen Y coordinate. */
static int to_sy(GuiState *g, int iy) { return g->disp_y + (int)(iy * g->disp_scale); }

/* -------------------------------------------------------------------------
 * Load image / model
 * ---------------------------------------------------------------------- */

/**
 * @brief Load a PNG image and create the display texture.
 *
 * Frees any previously loaded image, texture, and OCR results, then loads
 * the PNG at @p path via image_load_png().  An SDL texture is created from
 * the raw RGBA pixels (SDL_PIXELFORMAT_RGBA32) without copying them.
 * update_geometry() is called to recompute the display rectangle.
 *
 * @param g     Application state.
 * @param path  Path to the PNG file.
 */
static void gui_load_image(GuiState *g, const char *path)
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
        return;
    }

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
                 "Erreur texture: %s", SDL_GetError());
        g->status_ok = 0;
        return;
    }

    update_geometry(g);
    snprintf(g->status, sizeof(g->status),
             "Image chargée: %d×%d px", g->orig_img->width, g->orig_img->height);
    g->status_ok = 1;
}

/**
 * @brief Load CNN weights from a binary model file.
 *
 * Allocates @p g->net if necessary, then calls model_load().
 * On success @p g->model_buf already holds the path (set by the caller);
 * the status bar is updated to reflect the loaded filename.
 *
 * @param g     Application state.
 * @param path  Path to the .bin model file.
 */
static void gui_load_model(GuiState *g, const char *path)
{
    if (!g->net) {
        g->net = cnn_create();
        if (!g->net) {
            snprintf(g->status, sizeof(g->status), "Erreur: allocation CNN");
            g->status_ok = 0;
            return;
        }
    }
    if (model_load(g->net, path) != 0) {
        snprintf(g->status, sizeof(g->status),
                 "Erreur: chargement modèle '%s'", path);
        g->status_ok = 0;
        return;
    }
    const char *name = strrchr(path, '/');
    snprintf(g->status, sizeof(g->status),
             "Modèle chargé: %s", name ? name + 1 : path);
    g->status_ok = 1;
}

/* -------------------------------------------------------------------------
 * OCR pipeline
 * ---------------------------------------------------------------------- */

#define TTA_N 5

/**
 * @brief Run one CNN forward pass on a rectangular sub-region of a grayscale image.
 *
 * Copies the region [@p x1, @p x2) × [@p y1, @p y2) from @p gray into a
 * temporary Image, binarizes it locally, resizes to CNN_IMG_W × CNN_IMG_H,
 * then calls cnn_forward().  The resulting softmax probabilities are
 * **added** to @p probs (not overwritten), allowing TTA accumulation.
 *
 * @param gray        Full grayscale RGBA image (R=G=B=luminance).
 * @param x1,y1       Top-left of the region (clamped to image bounds).
 * @param x2,y2       Bottom-right exclusive (clamped to image bounds).
 * @param net         Trained CNN.
 * @param probs       Array of CNN_N_CLASSES floats; results are added here.
 */
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

/**
 * @brief Predict the letter in a grid cell using Test-Time Augmentation (TTA).
 *
 * Runs TTA_N forward passes centred on the bounding-box centre, each with a
 * small ±2 px spatial shift, averages the softmax outputs, and returns the
 * argmax class index (0='A' … 25='Z').
 *
 * The crop window is @p cell_size × @p cell_size (grid pitch) so that every
 * letter sees a consistent white border regardless of how tight the
 * connected-component bounding box is.  If @p cell_size is 0, a 35%-padding
 * heuristic is used instead.
 *
 * @param gray       Full grayscale image.
 * @param box        Tight bounding box returned by the segmenter.
 * @param cell_size  Grid pitch in pixels (pass 0 to use the padding fallback).
 * @param net        Trained CNN.
 * @return           Class index in [0, 25], or 0 on degenerate input.
 */
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

/**
 * @brief Run the full OCR pipeline and word search, then store results.
 *
 * Steps performed:
 *  1. Reload the image from @p g->image_buf and convert to grayscale.
 *  2. Build a binarized buffer for the segmenter (mean-threshold global binarization).
 *  3. Call segment_image() to detect letter bounding boxes.
 *  4. Estimate the grid pitch from the first/last cell centres.
 *  5. Call recognise_cell() for every cell (with TTA).
 *  6. Build a CharGrid and run solver_find() for each word in @p g->words_buf.
 *  7. Store the BoundingBox array in @p g->cells for highlight rendering.
 *
 * The status bar is updated at each major step so the render loop can show
 * progress messages.  The original image (@p g->orig_img) is never touched.
 *
 * @param g  Application state — must have orig_img and net set.
 */
static void gui_run_ocr(GuiState *g)
{
    if (!g->orig_img || !g->net) {
        snprintf(g->status, sizeof(g->status),
                 "Chargez une image et un modèle d'abord.");
        g->status_ok = 0;
        return;
    }

    if (g->cells) { free(g->cells); g->cells = NULL; g->n_cells = 0; }
    g->n_results = 0;

    Image *gray = image_load_png(g->image_buf);
    if (!gray) {
        snprintf(g->status, sizeof(g->status), "Erreur: rechargement image");
        g->status_ok = 0;
        return;
    }
    image_to_grayscale(gray);

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

    snprintf(g->status, sizeof(g->status), "Segmentation…");
    gui_render(g); /* show progress */

    SegmentResult *seg = segment_image(bin, gray->width, gray->height);
    free(bin);

    if (!seg || seg->count == 0) {
        image_free(gray);
        if (seg) segment_result_free(seg);
        snprintf(g->status, sizeof(g->status), "Erreur: aucune cellule détectée");
        g->status_ok = 0;
        return;
    }

    /* Grid pitch estimation */
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

    snprintf(g->status, sizeof(g->status),
             "Reconnaissance (%zu cellules)…", seg->count);
    gui_render(g);

    int *labels = malloc(seg->count * sizeof(int));
    if (!labels) { segment_result_free(seg); image_free(gray); return; }

    for (size_t i = 0; i < seg->count; i++)
        labels[i] = recognise_cell(gray, &seg->cells[i], cell_size, g->net);
    image_free(gray);

    /* Grid dimensions */
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

    g->cells   = malloc(seg->count * sizeof(BoundingBox));
    g->n_cells = seg->count;
    if (g->cells)
        memcpy(g->cells, seg->cells, seg->count * sizeof(BoundingBox));

    CharGrid *grid = grid_create(rows, cols);
    if (grid) grid_fill(grid, labels, seg->count);
    free(labels);
    segment_result_free(seg);

    if (!grid) {
        snprintf(g->status, sizeof(g->status), "Erreur: création grille");
        g->status_ok = 0;
        return;
    }

    printf("\nGrille (%d×%d):\n", rows, cols);
    grid_print(grid);

    /* Word search */
    g->n_results = 0;
    if (g->words_buf[0]) {
        char tmp[2048];
        memcpy(tmp, g->words_buf, sizeof(tmp));
        tmp[sizeof(tmp) - 1] = '\0';
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
                 "Grille %d×%d — %d/%d trouvé(s)",
                 rows, cols, found, g->n_results);
        g->status_ok = (found == g->n_results) ? 1 : -1;
    } else {
        snprintf(g->status, sizeof(g->status),
                 "Grille %d×%d — entrez des mots à chercher",
                 rows, cols);
        g->status_ok = 1;
    }
}

/* -------------------------------------------------------------------------
 * Word highlight
 * ---------------------------------------------------------------------- */

/**
 * @brief Overlay a semi-transparent red rectangle on each cell of a found word.
 *
 * Iterates from (start_r, start_c) to (end_r, end_c) using the direction
 * deltas derived from the WordResult, maps each cell index to a BoundingBox
 * in @p g->cells, converts the bounding-box centre to screen coordinates via
 * to_sx() / to_sy(), and draws a filled + outlined rectangle scaled by
 * @p g->disp_scale.
 *
 * @param g  Application state (provides cells, grid_cols, disp_scale, renderer).
 * @param r  Solver result for one word; no-op if @p r->found is 0.
 */
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

            int sx = to_sx(g, cx_img - half);
            int sy = to_sy(g, cy_img - half);
            int sw = (int)((half * 2) * g->disp_scale);
            int sh = (int)((half * 2) * g->disp_scale);
            if (sw < 4) sw = 4;
            if (sh < 4) sh = 4;

            SDL_SetRenderDrawBlendMode(g->renderer, SDL_BLENDMODE_BLEND);
            fill_rect(g->renderer, sx, sy, sw, sh, 220, 50, 50, 120);
            outline_rect(g->renderer, sx, sy, sw, sh, 255, 30, 30, 255);
        }
        if (row == r->end_r && col == r->end_c) break;
        row += dr; col += dc;
    }
}

/* -------------------------------------------------------------------------
 * Render
 * ---------------------------------------------------------------------- */

/**
 * @brief Composite and present one complete frame.
 *
 * Drawing order:
 *  1. Dark background.
 *  2. Control panel (rows 1–3: labels, input fields, buttons; status bar).
 *  3. Scaled image (if loaded), or a placeholder message.
 *  4. Word-highlight rectangles for every found WordResult.
 *
 * Called every ~16 ms from the event loop and also mid-OCR to show progress.
 *
 * @param g  Application state.
 */
static void gui_render(GuiState *g)
{
    SDL_SetRenderDrawColor(g->renderer, 36, 36, 40, 255);
    SDL_RenderClear(g->renderer);

    /* Panel background */
    fill_rect(g->renderer, 0, 0, WIN_W, PANEL_H, 50, 50, 56, 255);
    SDL_SetRenderDrawColor(g->renderer, 65, 65, 75, 255);
    SDL_RenderDrawLine(g->renderer, 0, PANEL_H - 1, WIN_W, PANEL_H - 1);

    /* Row 1: Image */
    if (g->font_sm)
        draw_text(g, g->font_sm, "Image:", 8,
                  ROW1_Y + (ROW_H - FONT_SIZE_SM) / 2, 160, 160, 170);
    draw_input(g, FOCUS_IMAGE, g->image_buf, "chemin vers l'image PNG…", ROW1_Y);
    draw_btn(g, 1, "Charger", BTN_X, ROW1_Y, 0);

    /* Row 2: Model */
    if (g->font_sm)
        draw_text(g, g->font_sm, "Modèle:", 8,
                  ROW2_Y + (ROW_H - FONT_SIZE_SM) / 2, 160, 160, 170);
    draw_input(g, FOCUS_MODEL, g->model_buf, "chemin vers le modèle .bin…", ROW2_Y);
    draw_btn(g, 2, "Charger", BTN_X, ROW2_Y, 0);

    /* Row 3: Words */
    if (g->font_sm)
        draw_text(g, g->font_sm, "Mots:", 8,
                  ROW3_Y + (ROW_H - FONT_SIZE_SM) / 2, 160, 160, 170);
    draw_input(g, FOCUS_WORDS, g->words_buf,
               "mots à chercher, séparés par des virgules…", ROW3_Y);
    draw_btn(g, 3, g->busy ? "…" : "Chercher", BTN_X, ROW3_Y, g->busy);

    /* Status */
    if (g->font_sm && g->status[0]) {
        Uint8 sr  = g->status_ok == 1 ?  80 : g->status_ok == 0 ? 220 : 160;
        Uint8 sgg = g->status_ok == 1 ? 200 : g->status_ok == 0 ?  80 : 160;
        Uint8 sb  = g->status_ok == 1 ?  80 : g->status_ok == 0 ?  80 : 175;
        draw_text(g, g->font_sm, g->status, 8, STATUS_Y, sr, sgg, sb);
    }

    /* Image area */
    if (g->orig_tex) {
        int dw = (int)(g->orig_img->width  * g->disp_scale);
        int dh = (int)(g->orig_img->height * g->disp_scale);
        SDL_Rect dst = {g->disp_x, g->disp_y, dw, dh};
        SDL_RenderCopy(g->renderer, g->orig_tex, NULL, &dst);

        for (int i = 0; i < g->n_results; i++)
            draw_word_highlight(g, &g->results[i]);
    } else {
        fill_rect(g->renderer, 0, PANEL_H, WIN_W, WIN_H - PANEL_H, 40, 40, 44, 255);
        if (g->font)
            draw_text(g, g->font,
                      "Saisissez un chemin d'image et cliquez Charger",
                      WIN_W / 2 - 230,
                      PANEL_H + (WIN_H - PANEL_H) / 2 - FONT_SIZE / 2,
                      75, 75, 85);
    }

    SDL_RenderPresent(g->renderer);
}

/* -------------------------------------------------------------------------
 * Events
 * ---------------------------------------------------------------------- */

/**
 * @brief Test whether a mouse position hits the action button on a given row.
 * @param mx,my  Mouse cursor position in window coordinates.
 * @param row_y  Top-left Y of the row to test.
 * @return       Non-zero if the button was hit.
 */
static int btn_hit(int mx, int my, int row_y)
{
    return mx >= BTN_X && mx < BTN_X + BTN_W &&
           my >= row_y && my < row_y + ROW_H;
}

/**
 * @brief Test whether a mouse position hits the text-input field on a given row.
 * @param mx,my  Mouse cursor position in window coordinates.
 * @param row_y  Top-left Y of the row to test.
 * @return       Non-zero if the field was hit.
 */
static int input_hit(int mx, int my, int row_y)
{
    return mx >= INPUT_X && mx < INPUT_X + INPUT_W &&
           my >= row_y && my < row_y + ROW_H;
}

/**
 * @brief Set keyboard focus to a field and start/stop SDL text input.
 * @param g          Application state.
 * @param new_focus  FOCUS_IMAGE, FOCUS_MODEL, FOCUS_WORDS, or FOCUS_NONE.
 */
static void set_focus(GuiState *g, int new_focus)
{
    g->focused = new_focus;
    if (new_focus != FOCUS_NONE)
        SDL_StartTextInput();
    else
        SDL_StopTextInput();
}

/**
 * @brief Handle a left mouse-button click.
 *
 * Updates focus based on which field was clicked, then triggers the
 * appropriate action if an action button was hit:
 *  - Row 1 button → gui_load_image()
 *  - Row 2 button → gui_load_model()
 *  - Row 3 button → gui_run_ocr()   (ignored while busy)
 *
 * @param g     Application state.
 * @param mx,my Mouse cursor position.
 */
static void handle_click(GuiState *g, int mx, int my)
{
    /* Focus management */
    if (input_hit(mx, my, ROW1_Y))      set_focus(g, FOCUS_IMAGE);
    else if (input_hit(mx, my, ROW2_Y)) set_focus(g, FOCUS_MODEL);
    else if (input_hit(mx, my, ROW3_Y)) set_focus(g, FOCUS_WORDS);
    else                                 set_focus(g, FOCUS_NONE);

    /* Button clicks */
    if (btn_hit(mx, my, ROW1_Y)) {
        if (g->image_buf[0]) gui_load_image(g, g->image_buf);
    } else if (btn_hit(mx, my, ROW2_Y)) {
        if (g->model_buf[0]) gui_load_model(g, g->model_buf);
    } else if (btn_hit(mx, my, ROW3_Y) && !g->busy) {
        g->busy = 1;
        gui_run_ocr(g);
        g->busy = 0;
    }
}

/**
 * @brief Update the hovered-button state on mouse motion.
 * @param g     Application state.
 * @param mx,my Current mouse cursor position.
 */
static void handle_motion(GuiState *g, int mx, int my)
{
    g->hovered_btn = 0;
    if      (btn_hit(mx, my, ROW1_Y)) g->hovered_btn = 1;
    else if (btn_hit(mx, my, ROW2_Y)) g->hovered_btn = 2;
    else if (btn_hit(mx, my, ROW3_Y)) g->hovered_btn = 3;
}

/**
 * @brief Return a pointer to the text buffer of the currently focused field.
 *
 * @param g    Application state.
 * @param cap  Output: byte capacity of the returned buffer.
 * @return     Pointer to the focused buffer, or NULL if no field is focused.
 */
static char *active_buf(GuiState *g, size_t *cap)
{
    switch (g->focused) {
    case FOCUS_IMAGE: *cap = sizeof(g->image_buf); return g->image_buf;
    case FOCUS_MODEL: *cap = sizeof(g->model_buf); return g->model_buf;
    case FOCUS_WORDS: *cap = sizeof(g->words_buf); return g->words_buf;
    default: *cap = 0; return NULL;
    }
}

/**
 * @brief Handle SDL_KEYDOWN events for the focused text field.
 *
 * Supported keys:
 *  - **Backspace** — delete the last character.
 *  - **Enter**     — validate the field (load file or run OCR).
 *  - **Ctrl+V**    — paste clipboard text (newlines stripped).
 *  - **Escape**    — clear focus.
 *  - **Tab**       — cycle focus to the next field.
 *
 * @param g    Application state.
 * @param key  SDL key symbol.
 */
static void handle_keydown(GuiState *g, SDL_Keycode key)
{
    size_t cap = 0;
    char *buf = active_buf(g, &cap);
    if (!buf) return;

    if (key == SDLK_BACKSPACE) {
        size_t len = strlen(buf);
        if (len > 0) buf[len - 1] = '\0';

    } else if (key == SDLK_RETURN || key == SDLK_KP_ENTER) {
        switch (g->focused) {
        case FOCUS_IMAGE:
            if (g->image_buf[0]) gui_load_image(g, g->image_buf);
            break;
        case FOCUS_MODEL:
            if (g->model_buf[0]) gui_load_model(g, g->model_buf);
            break;
        case FOCUS_WORDS:
            if (!g->busy) { g->busy = 1; gui_run_ocr(g); g->busy = 0; }
            break;
        }

    } else if (key == SDLK_v && (SDL_GetModState() & KMOD_CTRL)) {
        /* Paste from clipboard */
        char *clip = SDL_GetClipboardText();
        if (clip) {
            /* Strip newlines from pasted text */
            size_t cur = strlen(buf);
            for (char *p = clip; *p && cur < cap - 1; p++) {
                if (*p != '\n' && *p != '\r')
                    buf[cur++] = *p;
            }
            buf[cur] = '\0';
            SDL_free(clip);
        }

    } else if (key == SDLK_ESCAPE) {
        set_focus(g, FOCUS_NONE);

    } else if (key == SDLK_TAB) {
        /* Cycle fields */
        int next = g->focused + 1;
        if (next > FOCUS_WORDS) next = FOCUS_IMAGE;
        set_focus(g, next);
    }
}

/**
 * @brief Append SDL_TEXTINPUT characters to the focused field's buffer.
 *
 * SDL delivers printable characters via SDL_TEXTINPUT events (already
 * converted from key codes with correct locale/IME handling).  The text is
 * appended only if the buffer has room.
 *
 * @param g     Application state.
 * @param text  UTF-8 string from the SDL_TEXTINPUT event.
 */
static void handle_text_input(GuiState *g, const char *text)
{
    size_t cap = 0;
    char *buf = active_buf(g, &cap);
    if (!buf) return;
    size_t cur = strlen(buf);
    size_t add = strlen(text);
    if (cur + add < cap - 1)
        strncat(buf, text, cap - cur - 1);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

/**
 * @brief Entry point for the GUI binary.
 *
 * Initialises SDL2 and SDL2_ttf, creates the window and renderer, loads the
 * most recently modified model from @c models/ (or the path given via
 * @c --model), then enters the event loop.  The loop runs at ~60 fps and
 * dispatches events to the appropriate handler before calling gui_render().
 *
 * @param argc  Argument count.
 * @param argv  Argument vector.  Accepted options:
 *              @c --model @c \<path\> — explicit model file.
 * @return      0 on clean exit, 1 on SDL/TTF initialisation failure.
 */
int main(int argc, char **argv)
{
    const char *cli_model = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            cli_model = argv[++i];
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
        WIN_W, WIN_H, SDL_WINDOW_SHOWN
    );
    if (!g.window) {
        fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError());
        TTF_Quit(); SDL_Quit(); return 1;
    }

    g.renderer = SDL_CreateRenderer(
        g.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!g.renderer) {
        fprintf(stderr, "SDL_CreateRenderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(g.window);
        TTF_Quit(); SDL_Quit(); return 1;
    }
    SDL_SetRenderDrawBlendMode(g.renderer, SDL_BLENDMODE_BLEND);

    g.font    = find_font(FONT_SIZE);
    g.font_sm = find_font(FONT_SIZE_SM);
    if (!g.font)
        fprintf(stderr,
                "Avertissement: aucune police TTF trouvée.\n"
                "  Arch:   sudo pacman -S ttf-dejavu\n"
                "  Debian: sudo apt install fonts-dejavu\n");

    /* Auto-load latest model */
    g.net = cnn_create();
    if (g.net) {
        char mpath[512] = {0};
        if (cli_model) {
            strncpy(mpath, cli_model, sizeof(mpath) - 1);
        } else {
            model_find_latest(DEFAULT_MODEL_DIR, mpath, sizeof(mpath));
        }
        if (mpath[0]) {
            memcpy(g.model_buf, mpath, sizeof(g.model_buf));
            g.model_buf[sizeof(g.model_buf) - 1] = '\0';
            if (model_load(g.net, mpath) == 0) {
                const char *name = strrchr(mpath, '/');
                snprintf(g.status, sizeof(g.status),
                         "Modèle auto: %s", name ? name + 1 : mpath);
                g.status_ok = 1;
            }
        }
    }

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
