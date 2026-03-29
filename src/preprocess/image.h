/**
 * @file image.h
 * @brief Image loading, preprocessing, and normalisation.
 *
 * Uses libpng for PNG loading (no SDL2_image required).
 * SDL2 is used only for rotation rendering.
 *
 * Pipeline applied before CNN inference:
 *   load PNG → grayscale → binarize → (optional rotate) → resize 28×28 → float[784]
 *
 * Convention: black ink = 1.0, white background = 0.0.
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Opaque pixel buffer
 * ---------------------------------------------------------------------- */

/**
 * @brief Heap-allocated RGBA pixel buffer (row-major, 4 bytes per pixel).
 */
typedef struct {
    uint8_t *pixels;  /**< Raw RGBA data, row-major.            */
    int      width;   /**< Width in pixels.                     */
    int      height;  /**< Height in pixels.                    */
} Image;

/* -------------------------------------------------------------------------
 * Preprocessing parameters
 * ---------------------------------------------------------------------- */

/**
 * @brief Parameters for the full preprocessing pipeline.
 */
typedef struct {
    float rotation_deg; /**< Clockwise rotation angle in degrees (0 = none). */
    int   invert;       /**< If non-zero, invert after binarisation.          */
} PreprocessParams;

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/**
 * @brief Load a PNG file into an Image.
 *
 * @param path  Path to the PNG file.
 * @return      Heap-allocated Image on success, or NULL on error.
 *              Free with image_free().
 */
Image *image_load_png(const char *path);

/**
 * @brief Free an Image allocated by image_load_png().
 *
 * @param img  Image to free.  No-op if NULL.
 */
void image_free(Image *img);

/**
 * @brief Convert an Image to grayscale in-place.
 *
 * Uses ITU-R BT.601 luminance weights:
 *   L = 0.299·R + 0.587·G + 0.114·B
 *
 * After this call the R, G, and B channels all hold the luminance value.
 *
 * @param img  Image to convert.
 */
void image_to_grayscale(Image *img);

/**
 * @brief Binarise a grayscale Image in-place using a global threshold.
 *
 * The threshold is the mean pixel luminance of the image.
 * Pixels above the threshold → 255 (white/background).
 * Pixels at or below       → 0   (black/ink).
 *
 * @param img  Grayscale Image (R channel used as luminance).
 */
void image_binarize(Image *img);

/**
 * @brief Resize an Image to the given dimensions (nearest-neighbour).
 *
 * Returns a new Image; the original is not modified.
 *
 * @param img  Source Image.
 * @param w    Target width in pixels.
 * @param h    Target height in pixels.
 * @return     New Image on success, NULL on allocation failure.
 *             Free with image_free().
 */
Image *image_resize(const Image *img, int w, int h);

/**
 * @brief Rotate an Image by @p angle_deg degrees clockwise.
 *
 * Returns a new Image large enough to contain the full rotated content.
 * Background pixels are set to white (255, 255, 255, 255).
 *
 * @param img        Source Image.
 * @param angle_deg  Rotation angle in degrees (clockwise).
 * @return           New rotated Image, or NULL on failure.
 *                   Free with image_free().
 */
Image *image_rotate(const Image *img, float angle_deg);

/**
 * @brief Convert an Image to a flat normalised float array.
 *
 * Reads the R channel (expected to hold grayscale luminance).
 * Ink (R=0) → 1.0f, background (R=255) → 0.0f.
 *
 * @param img  Source Image (must be grayscale).
 * @param out  Output buffer of size img->width * img->height floats.
 */
void image_to_float(const Image *img, float *out);

/**
 * @brief Full pipeline: load PNG → preprocess → resize → float array.
 *
 * @param path   Path to the source PNG file.
 * @param p      Preprocessing parameters (may be NULL for defaults).
 * @param out    Output buffer of size @p out_h * @p out_w floats.
 * @param out_h  Target height (e.g. 28 for CNN input).
 * @param out_w  Target width  (e.g. 28 for CNN input).
 * @return       0 on success, -1 on error.
 */
int image_load_normalised(const char *path, const PreprocessParams *p,
                          float *out, int out_h, int out_w);

/**
 * @brief Save an Image as a PNG file.
 *
 * @param img   Image to save.
 * @param path  Destination file path.
 * @return      0 on success, -1 on error.
 */
int image_save_png(const Image *img, const char *path);

#endif /* IMAGE_H */
