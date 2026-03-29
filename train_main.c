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
#define MAX_EPOCHS        50
#define VAL_SPLIT         0.15f   /* fraction of data kept for validation */
#define ES_PATIENCE       5       /* epochs without improvement before stopping */
#define ES_MIN_DELTA      1e-4f   /* minimum val-loss improvement to count */

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
 * @brief Compute loss and accuracy on a slice of samples (no gradient update).
 *
 * @param net     Trained CNN.
 * @param samples Pointer to first sample.
 * @param n       Number of samples.
 * @param out_acc Filled with accuracy in [0, 1].
 * @return        Average cross-entropy loss.
 */
static double eval_loss(CNN *net, const Sample *samples, size_t n,
                        double *out_acc)
{
    double total = 0.0;
    int    correct = 0;

    for (size_t i = 0; i < n; i++) {
        cnn_forward(net, samples[i].pixels);
        total += (double)cnn_loss(net, samples[i].label);

        int pred = 0;
        float best = net->act.output[0];
        for (int k = 1; k < CNN_N_CLASSES; k++) {
            if (net->act.output[k] > best) {
                best = net->act.output[k];
                pred = k;
            }
        }
        if (pred == samples[i].label)
            correct++;
    }

    if (out_acc)
        *out_acc = (double)correct / (double)n;
    return total / (double)n;
}

/**
 * @brief Run the training loop with early stopping.
 *
 * The dataset is split into a training portion (1 - VAL_SPLIT) and a
 * validation portion (VAL_SPLIT) after an initial shuffle.
 *
 * Early stopping: training halts when validation loss has not improved by
 * more than ES_MIN_DELTA for ES_PATIENCE consecutive epochs.  The best
 * weights observed so far are restored before returning.
 *
 * @param net       Initialised CNN.
 * @param ds        Full dataset (will be shuffled in-place).
 * @param n_epochs  Maximum number of epochs.
 */
static void train_loop(CNN *net, Dataset *ds, int n_epochs)
{
    /* Split dataset after a full shuffle. */
    dataset_shuffle(ds);
    size_t val_size   = (size_t)(ds->size * VAL_SPLIT);
    size_t train_size = ds->size - val_size;
    const Sample *train_data = ds->data;
    const Sample *val_data   = ds->data + train_size;

    printf("Split: %zu train  /  %zu val\n\n", train_size, val_size);

    /* Snapshot buffer for best weights (early stopping). */
    CNNWeights best_weights = net->weights;
    double     best_val_loss = 1e30;
    int        patience_left = ES_PATIENCE;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        /* Shuffle only the training portion each epoch. */
        for (size_t i = train_size - 1; i > 0; i--) {
            size_t j = (size_t)rand() % (i + 1);
            Sample tmp         = ds->data[i];
            ds->data[i]        = ds->data[j];
            ds->data[j]        = tmp;
        }

        double train_loss = 0.0;
        int    correct    = 0;

        for (size_t i = 0; i < train_size; i++) {
            cnn_forward(net, train_data[i].pixels);
            train_loss += (double)cnn_loss(net, train_data[i].label);

            int pred = 0;
            float best = net->act.output[0];
            for (int k = 1; k < CNN_N_CLASSES; k++) {
                if (net->act.output[k] > best) {
                    best = net->act.output[k];
                    pred = k;
                }
            }
            if (pred == train_data[i].label)
                correct++;

            cnn_backward(net, train_data[i].label);

            if ((i + 1) % CNN_BATCH_SIZE == 0 || i + 1 == train_size)
                cnn_update(net);
        }

        train_loss /= (double)train_size;
        double train_acc = 100.0 * correct / (double)train_size;

        /* Validation pass. */
        double val_acc  = 0.0;
        double val_loss = eval_loss(net, val_data, val_size, &val_acc);
        val_acc        *= 100.0;

        printf("Epoch %2d/%d  train_loss=%.4f  acc=%.1f%%"
               "  |  val_loss=%.4f  val_acc=%.1f%%",
               epoch + 1, n_epochs,
               train_loss, train_acc,
               val_loss,   val_acc);

        /* Early stopping bookkeeping. */
        if (val_loss < best_val_loss - ES_MIN_DELTA) {
            best_val_loss = val_loss;
            best_weights  = net->weights;   /* snapshot */
            patience_left = ES_PATIENCE;
            printf("  [best]\n");
        } else {
            patience_left--;
            printf("  [patience %d/%d]\n", ES_PATIENCE - patience_left,
                   ES_PATIENCE);
            if (patience_left == 0) {
                printf("\nEarly stopping triggered (no improvement for %d"
                       " epochs).\n", ES_PATIENCE);
                break;
            }
        }
    }

    /* Restore best weights. */
    net->weights = best_weights;
    printf("Best val_loss=%.4f — weights restored.\n", best_val_loss);
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
    printf("  data dir    : %s\n", args.data_dir);
    printf("  output      : %s\n", args.output);
    printf("  threads     : %d\n", args.n_threads);
    printf("  max epochs  : %d\n", MAX_EPOCHS);
    printf("  batch sz    : %d\n", CNN_BATCH_SIZE);
    printf("  lr          : %.4f\n", (double)CNN_LR);
    printf("  val split   : %.0f%%\n", (double)(VAL_SPLIT * 100.0f));
    printf("  patience    : %d epochs\n", ES_PATIENCE);
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
