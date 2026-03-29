/**
 * @file train_main.c
 * @brief Entry point for the CNN training binary.
 *
 * Usage:
 * @code
 *   ./train [--data <dir>] [--output <file>] [-j<N>]
 * @endcode
 *
 * Options:
 *   --data   <dir>   Training data root directory (default: training_data/).
 *   --output <file>  Output model filename (default: models/model_<ts>.bin).
 *   -j<N>            Number of loader threads (default: 1).
 *
 * Example:
 * @code
 *   ./train --data training_data/ --output my_model.bin -j4
 * @endcode
 *
 * Exit codes:
 *   0  Success — model saved to the output path.
 *   1  Argument error.
 *   2  Dataset loading error.
 *   3  Model save error.
 */

#include "src/cnn/cnn.h"
#include "src/cnn/dataset.h"
#include "src/cnn/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* -------------------------------------------------------------------------
 * Default configuration
 * ---------------------------------------------------------------------- */

#define DEFAULT_DATA_DIR  "training_data/"
#define DEFAULT_MODEL_DIR "models/"
#define MAX_EPOCHS        20

/* -------------------------------------------------------------------------
 * CLI parsing
 * ---------------------------------------------------------------------- */

/**
 * @brief Parsed command-line options for the train binary.
 */
typedef struct {
    const char *data_dir;    /**< Path to training data root.     */
    char        output[512]; /**< Destination model file path.    */
    int         n_threads;   /**< Number of loader threads.       */
} TrainArgs;

/**
 * @brief Print usage information to stderr.
 *
 * @param prog  Program name (argv[0]).
 */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--data <dir>] [--output <file>] [-j<N>]\n"
            "\n"
            "  --data   <dir>   Training data root (default: %s)\n"
            "  --output <file>  Output model file  (default: %s/model_<ts>.bin)\n"
            "  -j<N>            Loader threads     (default: 1)\n",
            prog, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR);
}

/**
 * @brief Parse argv into a TrainArgs structure.
 *
 * @param argc     Argument count.
 * @param argv     Argument vector.
 * @param args     Output structure (caller-allocated).
 * @return         0 on success, -1 on parse error.
 */
static int parse_args(int argc, char **argv, TrainArgs *args)
{
    /* Defaults. */
    args->data_dir  = DEFAULT_DATA_DIR;
    args->output[0] = '\0';
    args->n_threads = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --data requires an argument\n");
                return -1;
            }
            args->data_dir = argv[++i];

        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --output requires an argument\n");
                return -1;
            }
            strncpy(args->output, argv[++i], sizeof(args->output) - 1);

        } else if (strncmp(argv[i], "-j", 2) == 0) {
            int n = atoi(argv[i] + 2);
            if (n < 1) {
                fprintf(stderr, "error: -j requires a positive integer\n");
                return -1;
            }
            args->n_threads = n;

        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            return -1;
        }
    }

    /* Generate default output path if not specified. */
    if (args->output[0] == '\0') {
        /* Ensure models/ directory exists. */
        mkdir(DEFAULT_MODEL_DIR, 0755);

        time_t ts = time(NULL);
        snprintf(args->output, sizeof(args->output),
                 "%smodel_%ld.bin", DEFAULT_MODEL_DIR, (long)ts);
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Training loop
 * ---------------------------------------------------------------------- */

/**
 * @brief Run the training loop for @p n_epochs epochs.
 *
 * Each epoch:
 *  1. Shuffle the dataset.
 *  2. Iterate over mini-batches of CNN_BATCH_SIZE samples.
 *  3. For each sample: forward pass → accumulate loss → backward pass.
 *  4. After each batch: apply cnn_update().
 *  5. Print epoch loss and accuracy.
 *
 * @param net       Initialised CNN.
 * @param ds        Training dataset.
 * @param n_epochs  Number of full passes over the dataset.
 */
static void train_loop(CNN *net, Dataset *ds, int n_epochs)
{
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        dataset_shuffle(ds);

        double total_loss = 0.0;
        int    correct    = 0;

        for (size_t i = 0; i < ds->size; i++) {
            const Sample *s = &ds->data[i];

            /* Forward pass. */
            cnn_forward(net, s->pixels);
            total_loss += (double)cnn_loss(net, s->label);

            /* Track accuracy. */
            int pred = 0;
            float best = net->act.output[0];
            for (int k = 1; k < CNN_N_CLASSES; k++) {
                if (net->act.output[k] > best) {
                    best = net->act.output[k];
                    pred = k;
                }
            }
            if (pred == s->label)
                correct++;

            /* Backward pass. */
            cnn_backward(net, s->label);

            /* Update weights at end of each mini-batch. */
            if ((i + 1) % CNN_BATCH_SIZE == 0 || i + 1 == ds->size)
                cnn_update(net);
        }

        double avg_loss = total_loss / (double)ds->size;
        double accuracy = 100.0 * correct / (double)ds->size;
        printf("Epoch %2d/%d  loss=%.4f  accuracy=%.2f%%\n",
               epoch + 1, n_epochs, avg_loss, accuracy);
    }
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    TrainArgs args;
    if (parse_args(argc, argv, &args) != 0) {
        usage(argv[0]);
        return 1;
    }

    printf("Training configuration:\n");
    printf("  data dir : %s\n", args.data_dir);
    printf("  output   : %s\n", args.output);
    printf("  threads  : %d\n", args.n_threads);
    printf("  epochs   : %d\n", MAX_EPOCHS);
    printf("  batch sz : %d\n", CNN_BATCH_SIZE);
    printf("  lr       : %.4f\n", (double)CNN_LR);
    printf("\n");

    /* Load dataset. */
    printf("Loading dataset from '%s'...\n", args.data_dir);
    Dataset *ds = dataset_load(args.data_dir, args.n_threads);
    if (!ds) {
        fprintf(stderr, "error: failed to load dataset from '%s'\n",
                args.data_dir);
        return 2;
    }
    dataset_print_info(ds);
    printf("\n");

    /* Initialise network. */
    srand((unsigned)time(NULL));
    CNN *net = cnn_create();
    if (!net) {
        fprintf(stderr, "error: failed to allocate CNN\n");
        dataset_free(ds);
        return 2;
    }

    /* Train. */
    printf("Starting training...\n");
    train_loop(net, ds, MAX_EPOCHS);
    printf("Training complete.\n\n");

    /* Save model. */
    printf("Saving model to '%s'...\n", args.output);
    if (model_save(net, args.output) != 0) {
        fprintf(stderr, "error: failed to save model\n");
        cnn_free(net);
        dataset_free(ds);
        return 3;
    }
    printf("Model saved successfully.\n");

    cnn_free(net);
    dataset_free(ds);
    return 0;
}
