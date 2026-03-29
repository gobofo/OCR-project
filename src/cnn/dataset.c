/**
 * @file dataset.c
 * @brief Training dataset loader with optional POSIX thread parallelism.
 */

#include "dataset.h"
#include "../preprocess/image.h"

#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/**
 * @brief Check whether @p filename ends with a known image extension.
 *
 * Accepted: .png, .jpg, .jpeg, .bmp (case-insensitive).
 *
 * @param filename File name (not full path).
 * @return         1 if the file is a recognised image, 0 otherwise.
 */
static int is_image_file(const char *filename)
{
    const char *ext = strrchr(filename, '.');
    if (!ext)
        return 0;
    return (strcasecmp(ext, ".png")  == 0 ||
            strcasecmp(ext, ".jpg")  == 0 ||
            strcasecmp(ext, ".jpeg") == 0 ||
            strcasecmp(ext, ".bmp")  == 0);
}

/* -------------------------------------------------------------------------
 * Worker thread context
 * ---------------------------------------------------------------------- */

/**
 * @brief Arguments passed to each loader thread.
 */
typedef struct {
    const char  *dir_path;  /**< Directory to scan for images.             */
    int          label;     /**< Class label (0–25) for all images in dir. */
    Sample      *buf;       /**< Pre-allocated buffer to write samples to.  */
    size_t       buf_cap;   /**< Capacity of @p buf (max samples to write). */
    size_t       n_loaded;  /**< Filled by thread: actual samples written.  */
} LoaderArgs;

/**
 * @brief POSIX thread worker: load all images from one class directory.
 *
 * Opens @p args->dir_path, iterates over image files, loads each one with
 * image_load_normalised() and writes the result to @p args->buf.
 *
 * @param arg  Pointer to a heap-allocated LoaderArgs (owned by caller).
 * @return     Always NULL (errors are printed to stderr).
 */
static void *loader_thread(void *arg)
{
    LoaderArgs *a = (LoaderArgs *)arg;
    a->n_loaded   = 0;

    DIR *d = opendir(a->dir_path);
    if (!d) {
        fprintf(stderr, "loader_thread: cannot open '%s'\n", a->dir_path);
        return NULL;
    }

    struct dirent *entry;
    while ((entry = readdir(d)) != NULL && a->n_loaded < a->buf_cap) {
        if (!is_image_file(entry->d_name))
            continue;

        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", a->dir_path, entry->d_name);

        Sample *s = &a->buf[a->n_loaded];
        int     rc = image_load_normalised(full, NULL, s->pixels,
                                           CNN_IMG_H, CNN_IMG_W);
        if (rc != 0) {
            fprintf(stderr, "loader_thread: failed to load '%s'\n", full);
            continue;
        }

        s->label = a->label;
        a->n_loaded++;
    }

    closedir(d);
    return NULL;
}

/* -------------------------------------------------------------------------
 * Count images in a directory (used to pre-allocate buffers)
 * ---------------------------------------------------------------------- */

/**
 * @brief Count the number of image files in a directory.
 *
 * @param dir_path Directory path.
 * @return         Number of image files (0 if directory cannot be opened).
 */
static size_t count_images(const char *dir_path)
{
    DIR *d = opendir(dir_path);
    if (!d)
        return 0;

    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL)
        if (is_image_file(entry->d_name))
            count++;

    closedir(d);
    return count;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

Dataset *dataset_load(const char *root_dir, int n_threads)
{
    Dataset *ds = calloc(1, sizeof(Dataset));
    if (!ds)
        return NULL;

    /* Clamp thread count. */
    if (n_threads < 1)
        n_threads = 1;
    if (n_threads > CNN_N_CLASSES)
        n_threads = CNN_N_CLASSES;

    /* Build per-class directory paths and count images. */
    char   class_dirs[CNN_N_CLASSES][512];
    size_t class_counts[CNN_N_CLASSES];
    size_t total = 0;

    for (int c = 0; c < CNN_N_CLASSES; c++) {
        snprintf(class_dirs[c], sizeof(class_dirs[c]),
                 "%s/%c", root_dir, 'A' + c);
        class_counts[c] = count_images(class_dirs[c]);
        total += class_counts[c];
    }

    if (total == 0) {
        fprintf(stderr,
                "dataset_load: no images found under '%s'\n", root_dir);
        free(ds);
        return NULL;
    }

    /* Allocate the full sample buffer upfront. */
    ds->data = malloc(total * sizeof(Sample));
    if (!ds->data) {
        fprintf(stderr, "dataset_load: out of memory (%zu samples)\n", total);
        free(ds);
        return NULL;
    }
    ds->cap  = total;
    ds->size = 0;

    /* Set up per-class loader args and assign buffer slices. */
    LoaderArgs args[CNN_N_CLASSES];
    size_t offset = 0;
    for (int c = 0; c < CNN_N_CLASSES; c++) {
        args[c].dir_path = class_dirs[c];
        args[c].label    = c;
        args[c].buf      = ds->data + offset;
        args[c].buf_cap  = class_counts[c];
        args[c].n_loaded = 0;
        offset          += class_counts[c];
    }

    /* Launch threads in batches of n_threads. */
    pthread_t threads[CNN_N_CLASSES];
    int       c = 0;

    while (c < CNN_N_CLASSES) {
        int batch = n_threads;
        if (c + batch > CNN_N_CLASSES)
            batch = CNN_N_CLASSES - c;

        /* Start batch. */
        for (int i = 0; i < batch; i++)
            pthread_create(&threads[i], NULL, loader_thread, &args[c + i]);

        /* Wait for batch to complete. */
        for (int i = 0; i < batch; i++) {
            pthread_join(threads[i], NULL);
            ds->size += args[c + i].n_loaded;
        }

        c += batch;
    }

    printf("dataset_load: loaded %zu samples from '%s'\n", ds->size, root_dir);
    return ds;
}

void dataset_shuffle(Dataset *ds)
{
    if (!ds || ds->size < 2)
        return;

    for (size_t i = ds->size - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        /* Swap samples[i] and samples[j]. */
        Sample tmp  = ds->data[i];
        ds->data[i] = ds->data[j];
        ds->data[j] = tmp;
    }
}

void dataset_free(Dataset *ds)
{
    if (!ds)
        return;
    free(ds->data);
    free(ds);
}

void dataset_print_info(const Dataset *ds)
{
    if (!ds) {
        printf("dataset_print_info: NULL dataset\n");
        return;
    }

    int counts[CNN_N_CLASSES] = {0};
    for (size_t i = 0; i < ds->size; i++)
        counts[ds->data[i].label]++;

    printf("Dataset: %zu samples\n", ds->size);
    for (int c = 0; c < CNN_N_CLASSES; c++)
        printf("  %c: %d\n", 'A' + c, counts[c]);
}
