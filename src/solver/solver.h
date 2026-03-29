/**
 * @file solver.h
 * @brief Crossword word-search solver: find words in a 2-D character grid
 *        in all 8 directions.
 *
 * The character grid is built from the CNN's predictions after segmentation:
 * each cell in the crossword image yields one predicted character (A–Z).
 * The solver then searches for user-supplied words in the resulting grid.
 *
 * Directions (clockwise from East):
 *   RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT, UP, UP_RIGHT
 */

#ifndef SOLVER_H
#define SOLVER_H

#include <stddef.h>

/* -------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------- */

/**
 * @brief A rectangular character grid.
 */
typedef struct {
    char  *cells;  /**< Flat array [row * cols + col] of uppercase letters. */
    int    rows;   /**< Number of rows.                                      */
    int    cols;   /**< Number of columns.                                   */
} CharGrid;

/**
 * @brief Result of a single word search.
 */
typedef struct {
    int found;    /**< 1 if the word was found, 0 otherwise.   */
    int start_r;  /**< Row of the first letter (0-based).      */
    int start_c;  /**< Column of the first letter (0-based).   */
    int end_r;    /**< Row of the last letter (0-based).       */
    int end_c;    /**< Column of the last letter (0-based).    */
    int dir;      /**< Direction index (0–7, see solver_dir_name()). */
} WordResult;

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/**
 * @brief Allocate a CharGrid of the given dimensions, filled with spaces.
 *
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @return      Heap-allocated CharGrid, or NULL on failure.
 *              Free with grid_free().
 */
CharGrid *grid_create(int rows, int cols);

/**
 * @brief Fill a CharGrid from an array of predicted character indices.
 *
 * @param grid    Target grid (must already be allocated with the right size).
 * @param labels  Array of class indices in [0, 25]; labels[r*cols + c] gives
 *                the index for cell (r, c).  0='A', 25='Z'.
 * @param count   Number of elements in @p labels (must equal rows * cols).
 */
void grid_fill(CharGrid *grid, const int *labels, size_t count);

/**
 * @brief Read a CharGrid from a plain-text file.
 *
 * File format (first line: dimensions, then one row per line):
 * @code
 *   rows cols
 *   ABCDE
 *   FGHIJ
 *   …
 * @endcode
 *
 * @param path  Path to the grid file.
 * @return      Heap-allocated CharGrid on success, NULL on error.
 *              Free with grid_free().
 */
CharGrid *grid_load(const char *path);

/**
 * @brief Print a CharGrid to stdout (for debugging / verbose mode).
 *
 * @param grid  Grid to display.
 */
void grid_print(const CharGrid *grid);

/**
 * @brief Free a CharGrid.
 *
 * @param grid  Grid to free.  No-op if NULL.
 */
void grid_free(CharGrid *grid);

/**
 * @brief Search for @p word in @p grid in all 8 directions.
 *
 * The search is case-insensitive; the grid and word are both converted to
 * uppercase before comparison.
 *
 * @param grid  Character grid to search in.
 * @param word  Null-terminated string to search for (A–Z, case-insensitive).
 * @return      WordResult with found=1 and coordinates if found,
 *              or found=0 if not present.
 */
WordResult solver_find(const CharGrid *grid, const char *word);

/**
 * @brief Return the human-readable name of direction index @p dir.
 *
 * Directions are numbered 0–7:
 *   0=RIGHT, 1=DOWN_RIGHT, 2=DOWN, 3=DOWN_LEFT,
 *   4=LEFT,  5=UP_LEFT,    6=UP,   7=UP_RIGHT
 *
 * @param dir  Direction index in [0, 7].
 * @return     Constant string (e.g. "RIGHT"), or "UNKNOWN" if out of range.
 */
const char *solver_dir_name(int dir);

#endif /* SOLVER_H */
