/**
 * @file image.c
 * @brief Image preprocessing pipeline using libpng (no SDL2_image required).
 */

#include "image.h"

#include <math.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Allocation helpers
 * ---------------------------------------------------------------------- */

/**
 * @brief Allocate an Image of given dimensions with a zeroed pixel buffer.
 *
 * @param w  Width in pixels.
 * @param h  Height in pixels.
 * @return   Heap-allocated Image, or NULL on failure.
 */
static Image *image_alloc(int w, int h)
{
    Image *img = malloc(sizeof(Image));
    if (!img)
        return NULL;
    img->pixels = calloc((size_t)w * h, 4);  /* 4 bytes RGBA per pixel */
    if (!img->pixels) {
        free(img);
        return NULL;
    }
    img->width  = w;
    img->height = h;
    return img;
}

void image_free(Image *img)
{
    if (!img)
        return;
    free(img->pixels);
    free(img);
}

/* -------------------------------------------------------------------------
 * PNG loading via libpng
 * ---------------------------------------------------------------------- */

Image *image_load_png(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "image_load_png: cannot open '%s'\n", path);
        return NULL;
    }

    /* Verify PNG signature. */
    uint8_t sig[8];
    if (fread(sig, 1, 8, fp) != 8 || !png_check_sig(sig, 8)) {
        fprintf(stderr, "image_load_png: '%s' is not a valid PNG\n", path);
        fclose(fp);
        return NULL;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                  NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return NULL; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    int width     = (int)png_get_image_width(png_ptr, info_ptr);
    int height    = (int)png_get_image_height(png_ptr, info_ptr);
    int color     = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    /* Normalise to 8-bit RGBA. */
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);
    if (color == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);
    if (color == PNG_COLOR_TYPE_RGB  ||
        color == PNG_COLOR_TYPE_GRAY ||
        color == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    if (color == PNG_COLOR_TYPE_GRAY ||
        color == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    Image *img = image_alloc(width, height);
    if (!img) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    /* Build row pointer array then read. */
    png_bytep *rows = malloc((size_t)height * sizeof(png_bytep));
    if (!rows) {
        image_free(img);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    for (int y = 0; y < height; y++)
        rows[y] = img->pixels + y * width * 4;

    png_read_image(png_ptr, rows);
    free(rows);

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return img;
}

/* -------------------------------------------------------------------------
 * PNG saving via libpng
 * ---------------------------------------------------------------------- */

int image_save_png(const Image *img, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "image_save_png: cannot open '%s' for writing\n", path);
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                   NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); return -1; }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return -1;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return -1;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr,
                 (png_uint_32)img->width, (png_uint_32)img->height,
                 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    for (int y = 0; y < img->height; y++)
        png_write_row(png_ptr, img->pixels + y * img->width * 4);

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return 0;
}

/* -------------------------------------------------------------------------
 * Grayscale
 * ---------------------------------------------------------------------- */

void image_to_grayscale(Image *img)
{
    uint8_t *p = img->pixels;
    int      n = img->width * img->height;
    for (int i = 0; i < n; i++, p += 4) {
        uint8_t lum = (uint8_t)(0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]);
        p[0] = p[1] = p[2] = lum;
        /* p[3] (alpha) unchanged */
    }
}

/* -------------------------------------------------------------------------
 * Binarisation
 * ---------------------------------------------------------------------- */

void image_binarize(Image *img)
{
    /* Compute mean luminance (R channel = grayscale). */
    long    sum  = 0;
    long    n    = (long)img->width * img->height;
    uint8_t *p   = img->pixels;

    for (long i = 0; i < n; i++, p += 4)
        sum += p[0];

    uint8_t threshold = (uint8_t)(sum / n);

    p = img->pixels;
    for (long i = 0; i < n; i++, p += 4) {
        uint8_t val = (p[0] > threshold) ? 255 : 0;
        p[0] = p[1] = p[2] = val;
    }
}

/* -------------------------------------------------------------------------
 * Resize (nearest-neighbour)
 * ---------------------------------------------------------------------- */

Image *image_resize(const Image *src, int w, int h)
{
    Image *dst = image_alloc(w, h);
    if (!dst)
        return NULL;

    float x_scale = (float)src->width  / (float)w;
    float y_scale = (float)src->height / (float)h;

    for (int dy = 0; dy < h; dy++) {
        int sy = (int)((float)dy * y_scale);
        if (sy >= src->height) sy = src->height - 1;

        for (int dx = 0; dx < w; dx++) {
            int sx = (int)((float)dx * x_scale);
            if (sx >= src->width) sx = src->width - 1;

            uint8_t *s = src->pixels + (sy * src->width + sx) * 4;
            uint8_t *d = dst->pixels + (dy * w + dx) * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }

    return dst;
}

/* -------------------------------------------------------------------------
 * Rotation (inverse mapping, nearest-neighbour)
 * ---------------------------------------------------------------------- */

Image *image_rotate(const Image *src, float angle_deg)
{
    if (angle_deg == 0.0f) {
        /* Return a copy. */
        Image *copy = image_alloc(src->width, src->height);
        if (copy)
            memcpy(copy->pixels, src->pixels,
                   (size_t)src->width * src->height * 4);
        return copy;
    }

    float rad   = angle_deg * (float)M_PI / 180.0f;
    float cos_a = cosf(rad);
    float sin_a = sinf(rad);

    /* Compute bounding box of rotated corners. */
    float w = (float)src->width, h = (float)src->height;
    float cx_s = w / 2.0f, cy_s = h / 2.0f;

    float corners_x[4] = {0.0f, w, 0.0f, w};
    float corners_y[4] = {0.0f, 0.0f, h, h};
    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;

    for (int i = 0; i < 4; i++) {
        float rx = (corners_x[i] - cx_s) * cos_a - (corners_y[i] - cy_s) * sin_a;
        float ry = (corners_x[i] - cx_s) * sin_a + (corners_y[i] - cy_s) * cos_a;
        if (rx < min_x) min_x = rx;
        if (rx > max_x) max_x = rx;
        if (ry < min_y) min_y = ry;
        if (ry > max_y) max_y = ry;
    }

    int new_w = (int)ceilf(max_x - min_x);
    int new_h = (int)ceilf(max_y - min_y);
    float cx_d = (float)new_w / 2.0f;
    float cy_d = (float)new_h / 2.0f;

    Image *dst = image_alloc(new_w, new_h);
    if (!dst)
        return NULL;

    /* Fill with white background. */
    memset(dst->pixels, 255, (size_t)new_w * new_h * 4);

    /* Inverse rotation: for each destination pixel, sample the source. */
    for (int dy = 0; dy < new_h; dy++) {
        for (int dx = 0; dx < new_w; dx++) {
            float tx = dx - cx_d;
            float ty = dy - cy_d;
            /* Inverse rotation = negate angle → transpose of rotation matrix. */
            float sx =  tx * cos_a + ty * sin_a + cx_s;
            float sy = -tx * sin_a + ty * cos_a + cy_s;

            int isx = (int)roundf(sx);
            int isy = (int)roundf(sy);
            if (isx < 0 || isx >= src->width || isy < 0 || isy >= src->height)
                continue;

            uint8_t *s = src->pixels + (isy * src->width + isx) * 4;
            uint8_t *d = dst->pixels + (dy * new_w + dx) * 4;
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
    }

    return dst;
}

/* -------------------------------------------------------------------------
 * Conversion to float
 * ---------------------------------------------------------------------- */

void image_to_float(const Image *img, float *out)
{
    int      n = img->width * img->height;
    uint8_t *p = img->pixels;
    for (int i = 0; i < n; i++, p += 4)
        out[i] = 1.0f - (float)p[0] / 255.0f;  /* ink=1.0, background=0.0 */
}

/* -------------------------------------------------------------------------
 * High-level pipeline
 * ---------------------------------------------------------------------- */

int image_load_normalised(const char *path, const PreprocessParams *p,
                          float *out, int out_h, int out_w)
{
    Image *img = image_load_png(path);
    if (!img)
        return -1;

    /* Optional rotation. */
    if (p && p->rotation_deg != 0.0f) {
        Image *rotated = image_rotate(img, p->rotation_deg);
        image_free(img);
        if (!rotated)
            return -1;
        img = rotated;
    }

    image_to_grayscale(img);
    image_binarize(img);

    /* Optional inversion. */
    if (p && p->invert) {
        int      n  = img->width * img->height;
        uint8_t *px = img->pixels;
        for (int i = 0; i < n; i++, px += 4)
            px[0] = px[1] = px[2] = 255 - px[0];
    }

    Image *resized = image_resize(img, out_w, out_h);
    image_free(img);
    if (!resized)
        return -1;

    image_to_float(resized, out);
    image_free(resized);
    return 0;
}
