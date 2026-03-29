/**
 * @file dataset.h
 * @brief Training dataset loader.
 *
 * Expects a root directory with one sub-directory per letter:
 *
 *   training_data/
 *     A/  image1.png  image2.png  …
 *     B/  …
 *     …
 *     Z/  …
 *
 * Each image is loaded with SDL2, converted to grayscale, and resized to
 * CNN_IMG_H × CNN_IMG_W.  The resulting pixel values are normalised to [0, 1].
 *
 * Parallelism: loading can be parallelised over POSIX threads; the number of
 * worker threads is controlled by the caller.
 */

#ifndef DATASET_H
#define DATASET_H

#include "cnn.h"
#include <stddef.h>

/**
 * @brief A single labelled training sample.
 */
typedef struct {
    /** Flat float array of CNN_IMG_H * CNN_IMG_W normalised pixels. */
    float pixels[CNN_IMG_H * CNN_IMG_W];
    /** Class index in [0, CNN_N_CLASSES): 0='A', 25='Z'. */
    int   label;
} Sample;

/**
 * @brief A collection of training samples.
 */
typedef struct {
    Sample *data;    /**< Heap-allocated array of samples.   */
    size_t  size;    /**< Number of valid samples in @p data. */
    size_t  cap;     /**< Allocated capacity of @p data.     */
} Dataset;

/**
 * @brief Load all images from @p root_dir into a Dataset.
 *
 * Opens sub-directories named A–Z.  For each PNG/JPG image found, loads it
 * with SDL2, converts to grayscale, resizes to 28×28, normalises to [0, 1],
 * and appends a Sample to the dataset.  Non-image files are silently skipped.
 *
 * @param root_dir   Path to the training data root (e.g. "training_data/").
 * @param n_threads  Number of POSIX loader threads (1 = single-threaded).
 * @return           Pointer to a heap-allocated Dataset, or NULL on error.
 *                   Free with dataset_free().
 *
 * @note Requires SDL2 and SDL2_image to be linked.
 */
Dataset *dataset_load(const char *root_dir, int n_threads);

/**
 * @brief Shuffle the samples in a Dataset in-place (Fisher-Yates).
 *
 * @param ds Dataset to shuffle.
 */
void dataset_shuffle(Dataset *ds);

/**
 * @brief Free all memory owned by a Dataset.
 *
 * @param ds Dataset returned by dataset_load().  No-op if NULL.
 */
void dataset_free(Dataset *ds);

/**
 * @brief Print a summary of the dataset to stdout.
 *
 * Shows the total number of samples and the per-class distribution.
 *
 * @param ds Dataset to describe.
 */
void dataset_print_info(const Dataset *ds);

#endif /* DATASET_H */
