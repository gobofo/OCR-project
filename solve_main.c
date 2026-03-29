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

/**
 * @brief Crop a cell bounding box from a pixel buffer, resize, and predict.
 *
 * Allocates a temporary Image for the cell, runs the full pipeline, then
 * calls cnn_predict().
 *
 * @param full_pixels  Full image pixel buffer (RGBA, row-major).
 * @param img_w        Full image width.
 * @param img_h        Full image height.
 * @param box          Bounding box of the cell to recognise.
 * @param net          Trained CNN.
 * @return             Predicted class index (0='A'…25='Z'), or -1 on error.
 */
static int recognise_cell(const unsigned char *full_pixels,
                           int img_w, int img_h,
                           const BoundingBox *box,
                           CNN *net)
{
    /* Build a sub-Image from the bounding box. */
    Image cell_img;
    cell_img.width  = box->w;
    cell_img.height = box->h;
    cell_img.pixels = malloc((size_t)box->w * box->h * 4);
    if (!cell_img.pixels)
        return -1;

    /* Copy pixels row by row. */
    for (int row = 0; row < box->h && (box->y + row) < img_h; row++) {
        const unsigned char *src = full_pixels
            + ((box->y + row) * img_w + box->x) * 4;
        unsigned char       *dst = cell_img.pixels + row * box->w * 4;
        int copy_w = box->w;
        if (box->x + copy_w > img_w)
            copy_w = img_w - box->x;
        memcpy(dst, src, (size_t)copy_w * 4);
    }

    /* Preprocess and resize to CNN input size. */
    Image *resized = image_resize(&cell_img, CNN_IMG_W, CNN_IMG_H);
    free(cell_img.pixels);
    if (!resized)
        return -1;

    float pixels[CNN_IMG_H * CNN_IMG_W];
    image_to_float(resized, pixels);
    image_free(resized);

    return cnn_predict(net, pixels);
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
    image_binarize(img);

    /* Build a single-channel (R-only) byte buffer for the segmenter. */
    int n_pixels = img->width * img->height;
    unsigned char *gray_buf = malloc((size_t)n_pixels);
    if (!gray_buf) {
        image_free(img);
        cnn_free(net);
        return 3;
    }
    for (int i = 0; i < n_pixels; i++)
        gray_buf[i] = img->pixels[i * 4];   /* R channel = grayscale */

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

    /* 5. Recognise each cell. */
    int *labels = malloc(seg->count * sizeof(int));
    if (!labels) {
        segment_result_free(seg);
        image_free(img);
        cnn_free(net);
        return 4;
    }

    size_t n_cells = seg->count;
    for (size_t i = 0; i < n_cells; i++) {
        labels[i] = recognise_cell(img->pixels, img->width, img->height,
                                    &seg->cells[i], net);
        if (args.verbose && labels[i] >= 0)
            printf("  cell %3zu: '%c'\n", i, 'A' + labels[i]);
    }

    /* 6. Build character grid. */
    int rows = seg->rows > 0 ? seg->rows : 1;
    int cols = seg->cols > 0 ? seg->cols : (int)n_cells;

    /* If grid dimensions unknown, try to infer from cell count. */
    if (seg->rows == 0 || seg->cols == 0) {
        for (int s = (int)n_cells; s >= 1; s--) {
            if ((int)n_cells % s == 0) {
                rows = (int)n_cells / s;
                cols = s;
                break;
            }
        }
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
