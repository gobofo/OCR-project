/**
 * @file segment.c
 * @brief Letter segmentation via connected-component analysis and grid detection.
 */

#include "segment.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/** Minimum number of black pixels in a component to be considered a letter. */
#define MIN_COMPONENT_SIZE   40

/** Maximum number of black pixels (filters out large blobs / noise). */
#define MAX_COMPONENT_SIZE   4000

/** Minimum aspect ratio (w/h) of a letter bounding box. */
#define MIN_ASPECT_RATIO     0.15f

/** Maximum aspect ratio (w/h) of a letter bounding box. */
#define MAX_ASPECT_RATIO     6.0f

/* -------------------------------------------------------------------------
 * Internal: queue for iterative flood-fill
 * ---------------------------------------------------------------------- */

/**
 * @brief Simple heap-allocated FIFO queue for pixel coordinates.
 */
typedef struct {
    int    *data;   /**< Flat array of (x, y) pairs: data[2*i], data[2*i+1]. */
    size_t  head;   /**< Read index.                                          */
    size_t  tail;   /**< Write index (next free slot).                        */
    size_t  cap;    /**< Total allocated pairs.                               */
} Queue;

/**
 * @brief Allocate a Queue with initial capacity @p cap pixel pairs.
 *
 * @param cap  Initial capacity (number of (x,y) pairs).
 * @return     Initialised Queue, or {NULL,0,0,0} on failure.
 */
static Queue queue_create(size_t cap)
{
    Queue q;
    q.data = malloc(cap * 2 * sizeof(int));
    q.head = q.tail = 0;
    q.cap  = q.data ? cap : 0;
    return q;
}

/**
 * @brief Append a pixel coordinate to the queue, growing if needed.
 *
 * @param q  Queue to push to.
 * @param x  Column.
 * @param y  Row.
 * @return   0 on success, -1 on allocation failure.
 */
static int queue_push(Queue *q, int x, int y)
{
    if (q->tail >= q->cap) {
        size_t new_cap = q->cap * 2;
        int   *ptr     = realloc(q->data, new_cap * 2 * sizeof(int));
        if (!ptr) return -1;
        q->data = ptr;
        q->cap  = new_cap;
    }
    q->data[q->tail * 2]     = x;
    q->data[q->tail * 2 + 1] = y;
    q->tail++;
    return 0;
}

/**
 * @brief Pop the front element.  Caller must ensure queue is non-empty.
 *
 * @param q   Queue.
 * @param x   Receives column.
 * @param y   Receives row.
 */
static void queue_pop(Queue *q, int *x, int *y)
{
    *x = q->data[q->head * 2];
    *y = q->data[q->head * 2 + 1];
    q->head++;
}

/** @brief Return 1 if the queue has elements, 0 if empty. */
static int queue_empty(const Queue *q) { return q->head >= q->tail; }

/** @brief Free queue memory. */
static void queue_free(Queue *q) { free(q->data); q->data = NULL; }

/* -------------------------------------------------------------------------
 * Connected-component analysis
 * ---------------------------------------------------------------------- */

int segment_connected_components(const unsigned char *pixels,
                                  int width, int height,
                                  SegmentResult *out)
{
    out->cells = NULL;
    out->count = 0;
    out->cols  = 0;
    out->rows  = 0;

    /* Visited map: 0 = unvisited, 1 = visited. */
    unsigned char *visited = calloc((size_t)width * height, 1);
    if (!visited)
        return -1;

    /* Preallocate result array. */
    size_t       cap   = 64;
    BoundingBox *cells = malloc(cap * sizeof(BoundingBox));
    if (!cells) {
        free(visited);
        return -1;
    }

    int n_found = 0;

    /* 4-connectivity neighbours (dx, dy). */
    static const int dx[4] = {1, -1, 0,  0};
    static const int dy[4] = {0,  0, 1, -1};

    Queue q = queue_create(4096);
    if (!q.data) {
        free(cells);
        free(visited);
        return -1;
    }

    for (int start_y = 0; start_y < height; start_y++) {
        for (int start_x = 0; start_x < width; start_x++) {
            int idx = start_y * width + start_x;
            if (visited[idx] || pixels[idx] != 0)
                continue;   /* skip white pixels and already-seen pixels */

            /* BFS flood-fill from (start_x, start_y). */
            int x1 = start_x, x2 = start_x;
            int y1 = start_y, y2 = start_y;
            int size = 0;

            visited[idx] = 1;
            q.head = q.tail = 0;   /* reset queue */
            queue_push(&q, start_x, start_y);

            while (!queue_empty(&q)) {
                int cx, cy;
                queue_pop(&q, &cx, &cy);
                size++;

                if (cx < x1) x1 = cx;
                if (cx > x2) x2 = cx;
                if (cy < y1) y1 = cy;
                if (cy > y2) y2 = cy;

                for (int d = 0; d < 4; d++) {
                    int nx = cx + dx[d];
                    int ny = cy + dy[d];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;
                    int nidx = ny * width + nx;
                    if (!visited[nidx] && pixels[nidx] == 0) {
                        visited[nidx] = 1;
                        queue_push(&q, nx, ny);
                    }
                }
            }

            /* Filter by size and aspect ratio. */
            if (size < MIN_COMPONENT_SIZE || size > MAX_COMPONENT_SIZE)
                continue;

            int   bw = x2 - x1 + 1;
            int   bh = y2 - y1 + 1;
            float ar = (float)bw / (float)bh;
            if (ar < MIN_ASPECT_RATIO || ar > MAX_ASPECT_RATIO)
                continue;

            /* Grow result array if needed. */
            if ((size_t)n_found >= cap) {
                cap *= 2;
                BoundingBox *ptr = realloc(cells, cap * sizeof(BoundingBox));
                if (!ptr) {
                    free(cells);
                    free(visited);
                    queue_free(&q);
                    return -1;
                }
                cells = ptr;
            }

            cells[n_found].x = x1;
            cells[n_found].y = y1;
            cells[n_found].w = bw;
            cells[n_found].h = bh;
            n_found++;
        }
    }

    queue_free(&q);
    free(visited);

    out->cells = cells;
    out->count = (size_t)n_found;
    return n_found;
}

/* -------------------------------------------------------------------------
 * Grid-line detection (stub)
 * ---------------------------------------------------------------------- */

int segment_detect_grid(const unsigned char *pixels,
                         int width, int height, float min_span,
                         SegmentResult *out)
{
    /*
     * @note STUB — grid detection not yet implemented.
     *
     * Algorithm outline:
     *
     * 1. Horizontal projection: for each row y, count black pixels.
     *    A row is a "grid line" if its black pixel count ≥ min_span * width.
     *    Collect the y-positions of grid lines → y_lines[].
     *
     * 2. Vertical projection: same for columns.
     *    Collect x-positions → x_lines[].
     *
     * 3. Compute cell size: median gap between consecutive y_lines and
     *    x_lines.
     *
     * 4. Generate BoundingBox for each cell in the grid:
     *    for row r in 0..n_rows-1, col c in 0..n_cols-1:
     *      x = x_lines[c] + margin
     *      y = y_lines[r] + margin
     *      w = x_lines[c+1] - x_lines[c] - 2*margin
     *      h = y_lines[r+1] - y_lines[r] - 2*margin
     *
     * 5. Populate out->cells, out->count, out->rows, out->cols.
     *    Return 1 if ≥ 2 horizontal and 2 vertical lines were found.
     */
    (void)pixels;
    (void)width;
    (void)height;
    (void)min_span;
    (void)out;
    return 0;  /* not found / not implemented */
}

/* -------------------------------------------------------------------------
 * Reading-order sort
 * ---------------------------------------------------------------------- */

/**
 * @brief qsort comparator: sort BoundingBox by y then x.
 *
 * Boxes with y-centres within 10 pixels of each other are considered the
 * same row and sorted left-to-right.
 */
static int bbox_cmp(const void *a, const void *b)
{
    const BoundingBox *ba = (const BoundingBox *)a;
    const BoundingBox *bb = (const BoundingBox *)b;

    int cy_a = ba->y + ba->h / 2;
    int cy_b = bb->y + bb->h / 2;

    /* If centres are within 10px, treat as same row → sort by x. */
    if (abs(cy_a - cy_b) <= 10)
        return ba->x - bb->x;

    return cy_a - cy_b;
}

void segment_sort_reading_order(SegmentResult *res)
{
    if (!res || res->count == 0)
        return;
    qsort(res->cells, res->count, sizeof(BoundingBox), bbox_cmp);
}

/* -------------------------------------------------------------------------
 * Top-level dispatcher
 * ---------------------------------------------------------------------- */

SegmentResult *segment_image(const unsigned char *pixels,
                              int width, int height)
{
    SegmentResult *res = calloc(1, sizeof(SegmentResult));
    if (!res)
        return NULL;

    /* Try grid detection first (works best for clean crossword scans). */
    if (segment_detect_grid(pixels, width, height, 0.7f, res) && res->count > 0)
        return res;

    /* Fall back to connected-component analysis. */
    if (segment_connected_components(pixels, width, height, res) < 0) {
        free(res);
        return NULL;
    }

    segment_sort_reading_order(res);
    return res;
}

/* -------------------------------------------------------------------------
 * Cleanup
 * ---------------------------------------------------------------------- */

void segment_result_free(SegmentResult *res)
{
    if (!res)
        return;
    free(res->cells);
    free(res);
}
