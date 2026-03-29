/**
 * @file cnn.h
 * @brief Convolutional Neural Network for A–Z character recognition.
 *
 * Architecture:
 *   Input 28×28 (float, 0–1)
 *   → Conv2D  16 filters 3×3, stride 1, no padding  → 16×26×26
 *   → ReLU
 *   → MaxPool 2×2, stride 2                          → 16×13×13
 *   → Flatten                                        → 2704
 *   → Dense   2704 → 128
 *   → ReLU
 *   → Dense   128  → 26
 *   → Softmax                                        → P(A)…P(Z)
 *
 * Training: He initialisation, SGD with momentum, cross-entropy loss.
 */

#ifndef CNN_H
#define CNN_H

#include <stddef.h>

/* -------------------------------------------------------------------------
 * Architecture constants
 * ---------------------------------------------------------------------- */

/** Input image height and width (pixels). */
#define CNN_IMG_H   28
#define CNN_IMG_W   28

/** Convolution: number of filters and kernel size. */
#define CNN_N_FILTERS   16
#define CNN_KERNEL_H     3
#define CNN_KERNEL_W     3

/** Derived conv output size (no padding, stride 1). */
#define CNN_CONV_H  (CNN_IMG_H - CNN_KERNEL_H + 1)   /* 26 */
#define CNN_CONV_W  (CNN_IMG_W - CNN_KERNEL_W + 1)   /* 26 */

/** Max-pooling window and stride. */
#define CNN_POOL_H  2
#define CNN_POOL_W  2

/** Derived pool output size. */
#define CNN_POOL_OUT_H  (CNN_CONV_H / CNN_POOL_H)   /* 13 */
#define CNN_POOL_OUT_W  (CNN_CONV_W / CNN_POOL_W)   /* 13 */

/** Flattened size after pooling. */
#define CNN_FLAT_SIZE  (CNN_N_FILTERS * CNN_POOL_OUT_H * CNN_POOL_OUT_W)  /* 2704 */

/** Hidden dense layer size. */
#define CNN_HIDDEN_SIZE  128

/** Number of output classes (A–Z). */
#define CNN_N_CLASSES  26

/* -------------------------------------------------------------------------
 * Hyper-parameters
 * ---------------------------------------------------------------------- */

/** Learning rate for SGD. */
#define CNN_LR          0.001f

/** Momentum coefficient. */
#define CNN_MOMENTUM    0.9f

/** Training batch size. */
#define CNN_BATCH_SIZE  32

/* -------------------------------------------------------------------------
 * Data structures
 * ---------------------------------------------------------------------- */

/**
 * @brief Learnable parameters of the CNN.
 *
 * Laid out flat so that model_save() / model_load() can serialise them with
 * a single fwrite / fread call.
 */
typedef struct {
    /** Conv layer: kernels[filter][row][col]. */
    float kernels[CNN_N_FILTERS][CNN_KERNEL_H][CNN_KERNEL_W];
    /** Conv layer biases, one per filter. */
    float conv_bias[CNN_N_FILTERS];

    /** Dense layer 1 weights: W1[hidden][flat]. */
    float W1[CNN_HIDDEN_SIZE][CNN_FLAT_SIZE];
    /** Dense layer 1 biases. */
    float b1[CNN_HIDDEN_SIZE];

    /** Dense layer 2 weights: W2[class][hidden]. */
    float W2[CNN_N_CLASSES][CNN_HIDDEN_SIZE];
    /** Dense layer 2 biases. */
    float b2[CNN_N_CLASSES];
} CNNWeights;

/**
 * @brief Intermediate activations produced by the forward pass.
 *
 * Kept in memory during training so that backward() can reuse them.
 */
typedef struct {
    /** Normalised input image: input[row][col] ∈ [0, 1]. */
    float input[CNN_IMG_H][CNN_IMG_W];

    /** Conv output after ReLU: conv_out[filter][row][col]. */
    float conv_out[CNN_N_FILTERS][CNN_CONV_H][CNN_CONV_W];

    /** Pool output: pool_out[filter][row][col]. */
    float pool_out[CNN_N_FILTERS][CNN_POOL_OUT_H][CNN_POOL_OUT_W];

    /**
     * Row/col of the max element in each pooling window (for backprop).
     * pool_max_r[f][r][c] = row inside the 2×2 window that held the max.
     */
    int pool_max_r[CNN_N_FILTERS][CNN_POOL_OUT_H][CNN_POOL_OUT_W];
    int pool_max_c[CNN_N_FILTERS][CNN_POOL_OUT_H][CNN_POOL_OUT_W];

    /** Flattened activations after pooling. */
    float flat[CNN_FLAT_SIZE];

    /** Hidden layer pre-activation (before ReLU). */
    float z1[CNN_HIDDEN_SIZE];
    /** Hidden layer post-activation (after ReLU). */
    float h1[CNN_HIDDEN_SIZE];

    /** Output layer pre-softmax. */
    float z2[CNN_N_CLASSES];
    /** Final class probabilities (softmax output). */
    float output[CNN_N_CLASSES];
} CNNActivations;

/**
 * @brief Full CNN state: weights, gradients, momentum buffers, activations.
 */
typedef struct {
    CNNWeights  weights;   /**< Current learnable parameters.             */
    CNNWeights  grads;     /**< Accumulated gradients (reset each batch). */
    CNNWeights  velocity;  /**< Momentum velocity buffer.                 */
    CNNActivations act;    /**< Forward-pass activations.                 */
} CNN;

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/**
 * @brief Allocate and initialise a CNN with He-initialised weights.
 *
 * All conv kernels and dense weights are drawn from a zero-mean normal
 * distribution with variance 2/fan_in (He, 2015).  Biases are zeroed.
 *
 * @return Pointer to a heap-allocated CNN, or NULL on allocation failure.
 * @note   The caller must free the returned pointer with cnn_free().
 */
CNN *cnn_create(void);

/**
 * @brief Free all memory associated with a CNN.
 *
 * @param net Pointer returned by cnn_create().  No-op if NULL.
 */
void cnn_free(CNN *net);

/**
 * @brief Run a full forward pass and populate net->act.
 *
 * Stages: Conv2D → ReLU → MaxPool → Flatten → Dense → ReLU → Dense → Softmax.
 *
 * @param net    Initialised CNN.
 * @param image  Flat array of CNN_IMG_H * CNN_IMG_W normalised float pixels,
 *               row-major order.
 */
void cnn_forward(CNN *net, const float *image);

/**
 * @brief Compute gradients via backpropagation.
 *
 * Must be called after cnn_forward().  Accumulates gradients into net->grads.
 *
 * @param net    CNN after a forward pass.
 * @param label  True class index in [0, CNN_N_CLASSES).
 *
 * @note This is a stub: the full implementation is deferred.
 */
void cnn_backward(CNN *net, int label);

/**
 * @brief Apply one SGD-with-momentum update step to the weights.
 *
 * w_new = w - lr * (β * v + g)
 * where v is the momentum velocity and g is the gradient.
 *
 * @param net    CNN with filled net->grads from cnn_backward().
 * @note         Resets net->grads to zero after the update.
 */
void cnn_update(CNN *net);

/**
 * @brief Zero all gradient accumulators in net->grads.
 *
 * @param net CNN whose gradients should be cleared.
 */
void cnn_zero_grads(CNN *net);

/**
 * @brief Predict the most likely class for a single image.
 *
 * Runs cnn_forward() internally.
 *
 * @param net    Initialised CNN with trained weights.
 * @param image  Flat float array, same layout as cnn_forward().
 * @return       Class index in [0, CNN_N_CLASSES), i.e. 0 = 'A', 25 = 'Z'.
 */
int cnn_predict(CNN *net, const float *image);

/**
 * @brief Compute cross-entropy loss for the current forward-pass output.
 *
 * loss = -log(output[label])
 *
 * @param net   CNN after cnn_forward().
 * @param label True class index.
 * @return      Scalar cross-entropy loss (≥ 0).
 */
float cnn_loss(const CNN *net, int label);

#endif /* CNN_H */
