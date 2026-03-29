/**
 * @file model.h
 * @brief Binary serialisation and deserialisation of CNN weights.
 *
 * The model file format is a flat binary dump of the CNNWeights struct,
 * preceded by a 4-byte magic number and a 4-byte version field:
 *
 *   [magic: 4 bytes 'OCRC'] [version: uint32_t] [CNNWeights: sizeof(CNNWeights) bytes]
 *
 * This allows fast save/load with a single fwrite/fread call and easy
 * detection of incompatible files.
 */

#ifndef MODEL_H
#define MODEL_H

#include "cnn.h"
#include <stdint.h>

/** Magic bytes written at the start of every model file. */
#define MODEL_MAGIC    "OCRC"

/** Current model file format version. Increment when CNNWeights changes. */
#define MODEL_VERSION  ((uint32_t)1)

/**
 * @brief Save the weights of a trained CNN to a binary file.
 *
 * Creates or overwrites the file at @p path.  The directory must exist.
 *
 * @param net  CNN whose weights should be saved.  Must not be NULL.
 * @param path Destination file path (e.g. "models/my_model.bin").
 * @return     0 on success, -1 on I/O error (message printed to stderr).
 */
int model_save(const CNN *net, const char *path);

/**
 * @brief Load CNN weights from a binary file into @p net.
 *
 * Validates the magic number and version before reading.  If validation
 * fails, @p net is left unchanged.
 *
 * @param net  CNN to load weights into.  Must not be NULL.
 * @param path Source file path.
 * @return     0 on success, -1 on I/O error or format mismatch.
 */
int model_load(CNN *net, const char *path);

/**
 * @brief Find the most recently modified .bin file inside a directory.
 *
 * Iterates over all regular files with a ".bin" suffix in @p dir and
 * returns the one with the highest mtime.
 *
 * @param dir        Directory to search (e.g. "models/").
 * @param out_path   Buffer that receives the full path of the chosen file.
 * @param out_len    Size of @p out_path in bytes.
 * @return           0 if a file was found and written to @p out_path,
 *                  -1 if the directory is empty or cannot be opened.
 */
int model_find_latest(const char *dir, char *out_path, size_t out_len);

#endif /* MODEL_H */
