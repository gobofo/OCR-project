/**
 * @file solve_main.c
 * @brief Entry point for the OCR + crossword-solver binary.
 *
 * Usage:
 * @code
 *   ./solve --image <path> [--model <path>] [--words <word1,word2,...>] [-v]
 * @endcode
 *
 * Options:
 *   --image  <path>    (required) Input crossword image.
 *   --model  <path>    (optional) Model file to load.
 *                      If absent, picks the most recent .bin in models/.
 *   --words  <list>    Comma-separated list of words to search for.
 *   --verbose / -v     Print per-cell recognition details and grid.
 *
 * Pipeline:
 *  1. Load model.
 *  2. Load and preprocess the image (grayscale, binarize).
 *  3. Segment letter cells.
 *  4. Recognise each cell with the CNN.
 *  5. Build a character grid.
 *  6. Solve the word search (if --words is given).
 *
 * Exit codes:
 *   0  Success.
 *   1  Argument error.
 *   2  Model error (not found / cannot load).
 *   3  Image load error.
 *   4  Segmentation error.
 */

#include "src/cnn/cnn.h"
#include "src/cnn/model.h"
#include "src/preprocess/image.h"
#include "src/segment/segment.h"
#include "src/solver/solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MODEL_DIR "models/"

/* -------------------------------------------------------------------------
 * CLI parsing
 * ---------------------------------------------------------------------- */

/**
 * @brief Parsed command-line options for the solve binary.
 */
typedef struct {
    const char *image_path;   /**< Path to the input crossword image.  */
    char        model_path[512]; /**< Path to the model .bin file.     */
    char        words[2048];  /**< Comma-separated word list.           */
    int         verbose;      /**< Non-zero if verbose mode requested.  */
} SolveArgs;

/**
 * @brief Print usage information to stderr.
 *
 * @param prog  Program name (argv[0]).
 */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --image <path> [--model <path>] "
            "[--words <w1,w2,...>] [-v]\n"
            "\n"
            "  --image  <path>   Input crossword image (required)\n"
            "  --model  <path>   Model file (default: latest in %s)\n"
            "  --words  <list>   Comma-separated words to search for\n"
            "  --verbose / -v    Verbose output\n",
            prog, DEFAULT_MODEL_DIR);
}

/**
 * @brief Parse argv into a SolveArgs structure.
 *
 * @param argc  Argument count.
 * @param argv  Argument vector.
 * @param args  Output structure (caller-allocated and zeroed).
 * @return      0 on success, -1 on error.
 */
static int parse_args(int argc, char **argv, SolveArgs *args)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --image requires an argument\n");
                return -1;
            }
            args->image_path = argv[++i];

        } else if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --model requires an argument\n");
                return -1;
            }
            strncpy(args->model_path, argv[++i], sizeof(args->model_path) - 1);

        } else if (strcmp(argv[i], "--words") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --words requires an argument\n");
                return -1;
            }
            strncpy(args->words, argv[++i], sizeof(args->words) - 1);

        } else if (strcmp(argv[i], "--verbose") == 0 ||
                   strcmp(argv[i], "-v")        == 0) {
            args->verbose = 1;

        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return -1;
        }
    }

    if (!args->image_path) {
        fprintf(stderr, "error: --image is required\n");
        return -1;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Model resolution
 * ---------------------------------------------------------------------- */

/**
 * @brief Ensure args->model_path is filled: auto-detect if not specified.
 *
 * @param args  SolveArgs with possibly empty model_path.
 * @return      0 if model_path is valid, -1 if no model could be found.
 */
static int resolve_model(SolveArgs *args)
{
    if (args->model_path[0] != '\0')
        return 0;  /* already set by --model */

    if (model_find_latest(DEFAULT_MODEL_DIR,
                          args->model_path,
                          sizeof(args->model_path)) != 0) {
        fprintf(stderr,
                "error: no model specified and no .bin file found in '%s'\n"
                "       Train a model first with: ./train --data training_data/\n",
                DEFAULT_MODEL_DIR);
        return -1;
    }

    printf("Auto-selected model: %s\n", args->model_path);
    return 0;
}

/* -------------------------------------------------------------------------
 * Per-cell recognition
 * ---------------------------------------------------------------------- */

/** Number of shifted crops averaged for Test-Time Augmentation. */
#define TTA_N_CROPS  5

/**
 * @brief Run one forward pass on a region of the grayscale image.
 *
 * Extracts [x1,x2) × [y1,y2) from gray_img (RGBA, grayscale), binarizes
 * the region locally, resizes to 28×28 and runs cnn_forward().
 * The CNN output probabilities are added into @p probs_out.
 *
 * @param gray_img   Full grayscale image (RGBA).
 * @param x1 y1 x2 y2  Region bounds (clamped to image boundaries internally).
 * @param net        Trained CNN.
 * @param probs_out  Array of CNN_N_CLASSES floats — result is *added* here.
 */
static void forward_region(const Image *gray_img,
                            int x1, int y1, int x2, int y2,
                            CNN *net, float *probs_out)
{
    int img_w = gray_img->width;
    int img_h = gray_img->height;
    if (x1 < 0)    x1 = 0;
    if (y1 < 0)    y1 = 0;
    if (x2 > img_w) x2 = img_w;
    if (y2 > img_h) y2 = img_h;
    int cw = x2 - x1, ch = y2 - y1;
    if (cw <= 0 || ch <= 0) return;

    Image cell_img;
    cell_img.width  = cw;
    cell_img.height = ch;
    cell_img.pixels = malloc((size_t)cw * ch * 4);
    if (!cell_img.pixels) return;

    for (int row = 0; row < ch; row++) {
        const unsigned char *src = gray_img->pixels
            + ((y1 + row) * img_w + x1) * 4;
        memcpy(cell_img.pixels + row * cw * 4, src, (size_t)cw * 4);
    }

    image_binarize(&cell_img);

    Image *resized = image_resize(&cell_img, CNN_IMG_W, CNN_IMG_H);
    free(cell_img.pixels);
    if (!resized) return;

    float px[CNN_IMG_H * CNN_IMG_W];
    image_to_float(resized, px);
    image_free(resized);

    cnn_forward(net, px);
    for (int k = 0; k < CNN_N_CLASSES; k++)
        probs_out[k] += net->act.output[k];
}

/**
 * @brief Predict a cell using Test-Time Augmentation (TTA).
 *
 * Runs TTA_N_CROPS forward passes with slightly different crop origins
 * (the original crop + 4 shifts of ±shift pixels in x and y), averages
 * the softmax outputs, and returns the argmax class.
 *
 * The crop window is grid-aware: it is centred on the letter's bounding-box
 * centre and sized to @p cell_size × @p cell_size (the detected grid pitch).
 * This guarantees a consistent white border around the letter regardless of
 * how tight the connected-component bounding box is.
 *
 * @param gray_img   Full grayscale image.
 * @param box        Tight bounding box from segmentation.
 * @param cell_size  Full grid cell side length (pixels).  Pass 0 to fall back
 *                   to the padding-fraction heuristic.
 * @param net        Trained CNN.
 * @return           Best class index (0='A'…25='Z'), or -1 on error.
 */
static int recognise_cell(const Image *gray_img,
                           const BoundingBox *box,
                           int cell_size,
                           CNN *net)
{
    /* Centre of the letter bounding box. */
    int cx = box->x + box->w / 2;
    int cy = box->y + box->h / 2;

    /* Half-size of the crop window. */
    int half;
    if (cell_size > 0) {
        half = cell_size / 2;
    } else {
        /* Fallback: pad by 35% around the bbox. */
        int pad = (int)((box->w > box->h ? box->w : box->h) * 0.35f);
        if (pad < 3) pad = 3;
        half = box->w / 2 + pad;
    }

    /* TTA: original crop + 4 small shifts. */
    static const int shifts[TTA_N_CROPS][2] = {
        { 0,  0},
        {-2,  0},
        { 2,  0},
        { 0, -2},
        { 0,  2},
    };

    float probs[CNN_N_CLASSES] = {0};
    for (int t = 0; t < TTA_N_CROPS; t++) {
        int dx = shifts[t][0], dy = shifts[t][1];
        forward_region(gray_img,
                       cx - half + dx, cy - half + dy,
                       cx + half + dx, cy + half + dy,
                       net, probs);
    }

    int best = 0;
    for (int k = 1; k < CNN_N_CLASSES; k++)
        if (probs[k] > probs[best])
            best = k;
    return best;
}

/* -------------------------------------------------------------------------
 * Word search
 * ---------------------------------------------------------------------- */

/**
 * @brief Search for all comma-separated words in @p args->words.
 *
 * Prints each result to stdout in the format:
 *   WORD: (r0,c0) → (r1,c1)  DIRECTION
 * or:
 *   WORD: not found
 *
 * @param grid   Recognised character grid.
 * @param words  Comma-separated word list (modified in-place by strtok).
 */
static void search_words(const CharGrid *grid, char *words)
{
    char *tok = strtok(words, ",");
    while (tok) {
        /* Strip leading/trailing whitespace. */
        while (*tok == ' ') tok++;
        char *end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';

        WordResult r = solver_find(grid, tok);
        if (r.found) {
            printf("  %-20s (%d,%d) → (%d,%d)  %s\n",
                   tok, r.start_r, r.start_c,
                   r.end_r,   r.end_c,
                   solver_dir_name(r.dir));
        } else {
            printf("  %-20s not found\n", tok);
        }

        tok = strtok(NULL, ",");
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    SolveArgs args;
    memset(&args, 0, sizeof(args));

    if (parse_args(argc, argv, &args) != 0) {
        usage(argv[0]);
        return 1;
    }

    /* 1. Resolve model path. */
    if (resolve_model(&args) != 0)
        return 2;

    /* 2. Load model. */
    CNN *net = cnn_create();
    if (!net) {
        fprintf(stderr, "error: failed to allocate CNN\n");
        return 2;
    }

    if (args.verbose)
        printf("Loading model '%s'...\n", args.model_path);

    if (model_load(net, args.model_path) != 0) {
        cnn_free(net);
        return 2;
    }

    /* 3. Load and preprocess the full image. */
    if (args.verbose)
        printf("Loading image '%s'...\n", args.image_path);

    Image *img = image_load_png(args.image_path);
    if (!img) {
        fprintf(stderr, "error: cannot load image '%s'\n", args.image_path);
        cnn_free(net);
        return 3;
    }

    image_to_grayscale(img);

    /* Build a binarized single-channel buffer for the segmenter,
     * without modifying img so it stays as grayscale for per-cell
     * local binarization during recognition. */
    int n_pixels = img->width * img->height;
    unsigned char *gray_buf = malloc((size_t)n_pixels);
    if (!gray_buf) {
        image_free(img);
        cnn_free(net);
        return 3;
    }
    {
        /* Compute global threshold (mean luminance). */
        long sum = 0;
        for (int i = 0; i < n_pixels; i++)
            sum += img->pixels[i * 4];
        unsigned char thr = (unsigned char)(sum / n_pixels);
        for (int i = 0; i < n_pixels; i++)
            gray_buf[i] = (img->pixels[i * 4] > thr) ? 255 : 0;
    }

    /* 4. Segment. */
    if (args.verbose)
        printf("Segmenting image (%d×%d)...\n", img->width, img->height);

    SegmentResult *seg = segment_image(gray_buf, img->width, img->height);
    free(gray_buf);
    if (!seg || seg->count == 0) {
        fprintf(stderr, "error: segmentation found no letter cells\n");
        image_free(img);
        cnn_free(net);
        return 4;
    }

    if (args.verbose)
        printf("Found %zu letter cells.\n", seg->count);

    /* 5. Estimate grid cell size (pitch) from the sorted bounding boxes.
     *
     * For a regular grid the full cell size equals the image extent divided
     * by the number of rows/cols.  We estimate this from the actual bounding
     * boxes: compute the span between first and last letter centre in each
     * axis and divide by (count-1). */
    int cell_size = 0;
    {
        /* Count rows (reuse the same y-jump logic used below). */
        int row_count = 1;
        if (seg->count > 1) {
            int tol2 = seg->cells[0].h > 0 ? seg->cells[0].h * 3 / 5 : 20;
            int prev_cy2 = seg->cells[0].y + seg->cells[0].h / 2;
            for (size_t i = 1; i < seg->count; i++) {
                int cy = seg->cells[i].y + seg->cells[i].h / 2;
                if (cy - prev_cy2 > tol2) { row_count++; prev_cy2 = cy; }
            }
        }
        int col_count = (row_count > 0) ? (int)seg->count / row_count : 1;

        if (row_count > 1 && col_count > 1 && (size_t)(row_count * col_count) == seg->count) {
            /* Vertical pitch from first and last row centres. */
            int cy_first = seg->cells[0].y + seg->cells[0].h / 2;
            int cy_last  = seg->cells[(row_count-1)*col_count].y
                         + seg->cells[(row_count-1)*col_count].h / 2;
            int v_pitch  = (cy_last - cy_first) / (row_count - 1);

            /* Horizontal pitch from first row. */
            int cx_first = seg->cells[0].x + seg->cells[0].w / 2;
            int cx_last  = seg->cells[col_count-1].x + seg->cells[col_count-1].w / 2;
            int h_pitch  = (cx_last - cx_first) / (col_count - 1);

            cell_size = (v_pitch + h_pitch) / 2;
        }
        if (args.verbose)
            printf("Grid pitch: %dpx (%d×%d)\n", cell_size, row_count, col_count);
    }

    int *labels = malloc(seg->count * sizeof(int));
    if (!labels) {
        segment_result_free(seg);
        image_free(img);
        cnn_free(net);
        return 4;
    }

    size_t n_cells = seg->count;
    for (size_t i = 0; i < n_cells; i++) {
        labels[i] = recognise_cell(img, &seg->cells[i], cell_size, net);
        if (args.verbose && labels[i] >= 0)
            printf("  cell %3zu: '%c'\n", i, 'A' + labels[i]);
    }

    /* 6. Build character grid. */
    int rows, cols;

    if (seg->rows > 0 && seg->cols > 0) {
        rows = seg->rows;
        cols = seg->cols;
    } else {
        /* Infer rows by counting y-center jumps in the sorted cell list. */
        rows = 1;
        if (n_cells > 1) {
            int tol = seg->cells[0].h > 0 ? seg->cells[0].h * 3 / 5 : 20;
            int prev_cy = seg->cells[0].y + seg->cells[0].h / 2;
            for (size_t i = 1; i < n_cells; i++) {
                int cy = seg->cells[i].y + seg->cells[i].h / 2;
                if (cy - prev_cy > tol) {
                    rows++;
                    prev_cy = cy;
                }
            }
        }
        cols = (rows > 0) ? (int)n_cells / rows : (int)n_cells;
    }

    CharGrid *grid = grid_create(rows, cols);
    if (grid)
        grid_fill(grid, labels, n_cells);

    free(labels);
    segment_result_free(seg);
    image_free(img);

    /* Print grid. */
    printf("\nRecognised grid (%d×%d):\n", rows, cols);
    if (grid)
        grid_print(grid);

    /* 7. Word search. */
    if (args.words[0] != '\0' && grid) {
        printf("\nWord search results:\n");
        search_words(grid, args.words);
    }

    grid_free(grid);
    cnn_free(net);
    return 0;
}
