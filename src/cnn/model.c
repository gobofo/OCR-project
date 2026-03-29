/**
 * @file model.c
 * @brief Binary save/load of CNN weights and latest-model discovery.
 */

#include "model.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------
 * Save
 * ---------------------------------------------------------------------- */

int model_save(const CNN *net, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "model_save: cannot open '%s' for writing\n", path);
        return -1;
    }

    /* Write magic. */
    if (fwrite(MODEL_MAGIC, 1, 4, fp) != 4) {
        fprintf(stderr, "model_save: write error (magic)\n");
        fclose(fp);
        return -1;
    }

    /* Write version. */
    uint32_t version = MODEL_VERSION;
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "model_save: write error (version)\n");
        fclose(fp);
        return -1;
    }

    /* Write the entire weights struct in one shot. */
    if (fwrite(&net->weights, sizeof(CNNWeights), 1, fp) != 1) {
        fprintf(stderr, "model_save: write error (weights)\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/* -------------------------------------------------------------------------
 * Load
 * ---------------------------------------------------------------------- */

int model_load(CNN *net, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "model_load: cannot open '%s'\n", path);
        return -1;
    }

    /* Validate magic. */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, MODEL_MAGIC, 4) != 0) {
        fprintf(stderr, "model_load: '%s' is not a valid model file\n", path);
        fclose(fp);
        return -1;
    }

    /* Validate version. */
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "model_load: truncated file '%s'\n", path);
        fclose(fp);
        return -1;
    }
    if (version != MODEL_VERSION) {
        fprintf(stderr,
                "model_load: version mismatch in '%s' (file=%u, expected=%u)\n",
                path, version, MODEL_VERSION);
        fclose(fp);
        return -1;
    }

    /* Read weights. */
    if (fread(&net->weights, sizeof(CNNWeights), 1, fp) != 1) {
        fprintf(stderr, "model_load: truncated weights in '%s'\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/* -------------------------------------------------------------------------
 * Find latest model
 * ---------------------------------------------------------------------- */

int model_find_latest(const char *dir, char *out_path, size_t out_len)
{
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "model_find_latest: cannot open directory '%s'\n", dir);
        return -1;
    }

    time_t      best_mtime = -1;
    char        best_name[1024] = {0};
    struct dirent *entry;

    while ((entry = readdir(d)) != NULL) {
        /* Skip entries that don't end with ".bin". */
        size_t nlen = strlen(entry->d_name);
        if (nlen < 4 || strcmp(entry->d_name + nlen - 4, ".bin") != 0)
            continue;

        /* Build full path and stat it. */
        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir, entry->d_name);

        struct stat st;
        if (stat(full, &st) != 0)
            continue;
        if (!S_ISREG(st.st_mode))
            continue;

        if (st.st_mtime > best_mtime) {
            best_mtime = st.st_mtime;
            snprintf(best_name, sizeof(best_name), "%s", full);
        }
    }

    closedir(d);

    if (best_mtime < 0)
        return -1;  /* no .bin file found */

    strncpy(out_path, best_name, out_len - 1);
    out_path[out_len - 1] = '\0';
    return 0;
}
