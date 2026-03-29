/**
 * @file segment.h
 * @brief Letter segmentation: detect crossword grid and extract letter cells.
 *
 * Pipeline:
 *  1. Detect the crossword grid bounding box in the binarised image.
 *  2. Identify grid lines to compute cell size and origin.
 *  3. Return an ordered array of BoundingBox, one per cell, left-to-right,
 *     top-to-bottom.
 *  4. Each bounding box can then be cropped, resized, and fed to the CNN.
 *
 * Alternatively, for images with clearly separated letter blobs, the
 * connected-component analysis path extracts letter regions directly.
 */

#ifndef SEGMENT_H
#define SEGMENT_H

#include <stddef.h>

/* -------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------- */

/**
 * @brief Axis-aligned bounding box of one letter cell.
 */
typedef struct {
    int x;  /**< Left edge (column index, 0-based).  */
    int y;  /**< Top edge  (row index, 0-based).     */
    int w;  /**< Width  in pixels.                   */
    int h;  /**< Height in pixels.                   */
} BoundingBox;

/**
 * @brief Result of a grid detection / segmentation pass.
 */
typedef struct {
    BoundingBox *cells;  /**< Heap-allocated array of cell bounding boxes. */
    size_t       count;  /**< Number of cells.                             */
    int          cols;   /**< Grid width  in cells (0 if unknown).         */
    int          rows;   /**< Grid height in cells (0 if unknown).         */
} SegmentResult;

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/**
 * @brief Segment a binarised image into letter cell bounding boxes.
 *
 * Tries two strategies in order:
 *  1. Grid-line detection (for crossword-style images with visible grid).
 *  2. Connected-component analysis (for images without explicit grid lines).
 *
 * The returned SegmentResult::cells array is ordered left-to-right,
 * top-to-bottom.
 *
 * @param pixels   Flat grayscale pixel array (1 byte/pixel, 0=black, 255=white),
 *                 row-major, width × height bytes.
 * @param width    Image width in pixels.
 * @param height   Image height in pixels.
 * @return         Heap-allocated SegmentResult on success, NULL on error.
 *                 Free with segment_result_free().
 *
 * @note STUB: grid-line detection and connected-component paths are partially
 *       implemented; see internal comments for the full algorithm outline.
 */
SegmentResult *segment_image(const unsigned char *pixels,
                              int width, int height);

/**
 * @brief Detect grid lines and infer cell geometry.
 *
 * Finds horizontal and vertical runs of dark pixels spanning at least
 * @p min_span proportion of the image dimension to identify grid lines.
 * Returns regular grid cell positions if a consistent grid is detected.
 *
 * @param pixels    Grayscale pixel array (0=black, 255=white), row-major.
 * @param width     Image width.
 * @param height    Image height.
 * @param min_span  Minimum fraction of dimension a line must span [0, 1].
 * @param out       Pre-allocated SegmentResult to fill.
 * @return          1 if a regular grid was detected and @p out filled,
 *                  0 if no grid was found.
 *
 * @note STUB — full implementation deferred.
 */
int segment_detect_grid(const unsigned char *pixels,
                         int width, int height, float min_span,
                         SegmentResult *out);

/**
 * @brief Extract letter bounding boxes via connected-component analysis.
 *
 * Uses an iterative flood-fill (queue-based, no recursion) to group
 * connected black pixels into components.  Components whose size and
 * aspect ratio fall within the expected range for a letter are kept.
 *
 * @param pixels  Grayscale pixel array (0=black, 255=white), row-major.
 * @param width   Image width.
 * @param height  Image height.
 * @param out     Pre-allocated SegmentResult to fill.
 * @return        Number of components found (≥ 0), -1 on error.
 */
int segment_connected_components(const unsigned char *pixels,
                                  int width, int height,
                                  SegmentResult *out);

/**
 * @brief Sort a SegmentResult's cells in reading order (top-to-bottom,
 *        left-to-right).
 *
 * Cells are grouped into rows based on vertical overlap, then sorted
 * left-to-right within each row.
 *
 * @param res  SegmentResult to sort in-place.
 */
void segment_sort_reading_order(SegmentResult *res);

/**
 * @brief Free a SegmentResult returned by segment_image().
 *
 * @param res  Result to free.  No-op if NULL.
 */
void segment_result_free(SegmentResult *res);

#endif /* SEGMENT_H */
