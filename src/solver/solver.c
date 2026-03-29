/**
 * @file solver.c
 * @brief 8-direction crossword word-search solver.
 */

#include "solver.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Direction vectors
 * ---------------------------------------------------------------------- */

/** Number of search directions. */
#define N_DIRS 8

/**
 * @brief Row and column deltas for each of the 8 search directions.
 *
 * Indexed 0–7:
 *   0 = RIGHT      (dr= 0, dc=+1)
 *   1 = DOWN_RIGHT (dr=+1, dc=+1)
 *   2 = DOWN       (dr=+1, dc= 0)
 *   3 = DOWN_LEFT  (dr=+1, dc=-1)
 *   4 = LEFT       (dr= 0, dc=-1)
 *   5 = UP_LEFT    (dr=-1, dc=-1)
 *   6 = UP         (dr=-1, dc= 0)
 *   7 = UP_RIGHT   (dr=-1, dc=+1)
 */
static const int DIR_DR[N_DIRS] = { 0,  1,  1,  1,  0, -1, -1, -1};
static const int DIR_DC[N_DIRS] = { 1,  1,  0, -1, -1, -1,  0,  1};

static const char *DIR_NAMES[N_DIRS] = {
    "RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT",
    "LEFT",  "UP_LEFT",    "UP",   "UP_RIGHT"
};

/* -------------------------------------------------------------------------
 * Grid lifecycle
 * ---------------------------------------------------------------------- */

CharGrid *grid_create(int rows, int cols)
{
    CharGrid *g = malloc(sizeof(CharGrid));
    if (!g)
        return NULL;

    g->cells = malloc((size_t)rows * cols + 1);  /* +1 for safety */
    if (!g->cells) {
        free(g);
        return NULL;
    }
    memset(g->cells, ' ', (size_t)rows * cols);
    g->cells[rows * cols] = '\0';
    g->rows = rows;
    g->cols = cols;
    return g;
}

void grid_fill(CharGrid *grid, const int *labels, size_t count)
{
    size_t n = (size_t)grid->rows * grid->cols;
    if (count < n)
        n = count;
    for (size_t i = 0; i < n; i++)
        grid->cells[i] = (char)('A' + labels[i]);
}

CharGrid *grid_load(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "grid_load: cannot open '%s'\n", path);
        return NULL;
    }

    int rows, cols;
    if (fscanf(fp, "%d %d\n", &rows, &cols) != 2 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "grid_load: invalid header in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    CharGrid *g = grid_create(rows, cols);
    if (!g) {
        fclose(fp);
        return NULL;
    }

    char line[1024];
    for (int r = 0; r < rows; r++) {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "grid_load: unexpected EOF at row %d\n", r);
            grid_free(g);
            fclose(fp);
            return NULL;
        }
        /* Copy up to cols characters, uppercased. */
        for (int c = 0; c < cols && line[c] != '\0' && line[c] != '\n'; c++)
            g->cells[r * cols + c] = (char)toupper((unsigned char)line[c]);
    }

    fclose(fp);
    return g;
}

void grid_print(const CharGrid *g)
{
    if (!g) {
        printf("(null grid)\n");
        return;
    }
    for (int r = 0; r < g->rows; r++) {
        for (int c = 0; c < g->cols; c++)
            putchar(g->cells[r * g->cols + c]);
        putchar('\n');
    }
}

void grid_free(CharGrid *g)
{
    if (!g)
        return;
    free(g->cells);
    free(g);
}

/* -------------------------------------------------------------------------
 * Solver
 * ---------------------------------------------------------------------- */

/**
 * @brief Check whether @p word fits in @p grid starting at (r, c) going
 *        in direction @p dir.
 *
 * @param grid  Character grid.
 * @param word  Uppercase null-terminated search string.
 * @param r     Starting row.
 * @param c     Starting column.
 * @param dir   Direction index (0–7).
 * @return      1 if the word matches in that direction, 0 otherwise.
 */
static int try_direction(const CharGrid *grid, const char *word,
                          int r, int c, int dir)
{
    int dr  = DIR_DR[dir];
    int dc  = DIR_DC[dir];
    int len = (int)strlen(word);

    for (int i = 0; i < len; i++) {
        int nr = r + i * dr;
        int nc = c + i * dc;
        if (nr < 0 || nr >= grid->rows || nc < 0 || nc >= grid->cols)
            return 0;
        if (grid->cells[nr * grid->cols + nc] != word[i])
            return 0;
    }
    return 1;
}

WordResult solver_find(const CharGrid *grid, const char *word)
{
    WordResult result = {0, 0, 0, 0, 0, 0};

    if (!grid || !word || word[0] == '\0')
        return result;

    /* Build an uppercase copy of the word. */
    size_t wlen = strlen(word);
    char  *up   = malloc(wlen + 1);
    if (!up)
        return result;

    for (size_t i = 0; i <= wlen; i++)
        up[i] = (char)toupper((unsigned char)word[i]);

    /* Search every cell in every direction. */
    for (int r = 0; r < grid->rows && !result.found; r++) {
        for (int c = 0; c < grid->cols && !result.found; c++) {
            for (int d = 0; d < N_DIRS && !result.found; d++) {
                if (try_direction(grid, up, r, c, d)) {
                    result.found   = 1;
                    result.start_r = r;
                    result.start_c = c;
                    result.end_r   = r + (int)(wlen - 1) * DIR_DR[d];
                    result.end_c   = c + (int)(wlen - 1) * DIR_DC[d];
                    result.dir     = d;
                }
            }
        }
    }

    free(up);
    return result;
}

/* -------------------------------------------------------------------------
 * Utilities
 * ---------------------------------------------------------------------- */

const char *solver_dir_name(int dir)
{
    if (dir < 0 || dir >= N_DIRS)
        return "UNKNOWN";
    return DIR_NAMES[dir];
}
