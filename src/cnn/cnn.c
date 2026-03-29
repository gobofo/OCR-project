/**
 * @file cnn.c
 * @brief CNN implementation: initialisation, forward pass, backward pass,
 *        SGD-with-momentum update.
 */

#include "cnn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/**
 * @brief Box-Muller transform: sample one value from N(0, 1).
 *
 * Uses the standard two-uniform-samples formula.  Not thread-safe due to
 * static state; call from a single thread during initialisation.
 *
 * @return One sample from the standard normal distribution.
 */
static float randn(void)
{
    static int   have_spare = 0;
    static float spare;

    if (have_spare) {
        have_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    float mul = sqrtf(-2.0f * logf(s) / s);
    spare      = v * mul;
    have_spare = 1;
    return u * mul;
}

/**
 * @brief Apply ReLU element-wise: f(x) = max(0, x).
 *
 * @param x      Input/output array (modified in-place).
 * @param length Number of elements.
 */
static void relu_inplace(float *x, size_t length)
{
    for (size_t i = 0; i < length; i++)
        if (x[i] < 0.0f)
            x[i] = 0.0f;
}

/**
 * @brief Apply softmax in-place over an array of length n.
 *
 * Subtracts the maximum value before exponentiation for numerical stability.
 *
 * @param x      Input/output array (modified in-place).
 * @param length Number of elements.
 */
static void softmax_inplace(float *x, size_t length)
{
    float max_val = x[0];
    for (size_t i = 1; i < length; i++)
        if (x[i] > max_val)
            max_val = x[i];

    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (size_t i = 0; i < length; i++)
        x[i] /= sum;
}

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

CNN *cnn_create(void)
{
    CNN *net = calloc(1, sizeof(CNN));
    if (!net) {
        fprintf(stderr, "cnn_create: out of memory\n");
        return NULL;
    }

    /* He initialisation for conv kernels: std = sqrt(2 / fan_in).
     * fan_in for a conv filter = kernel_h * kernel_w * 1 input channel. */
    float conv_std = sqrtf(2.0f / (float)(CNN_KERNEL_H * CNN_KERNEL_W));
    for (int f = 0; f < CNN_N_FILTERS; f++)
        for (int r = 0; r < CNN_KERNEL_H; r++)
            for (int c = 0; c < CNN_KERNEL_W; c++)
                net->weights.kernels[f][r][c] = randn() * conv_std;

    /* He initialisation for Dense layer 1: fan_in = CNN_FLAT_SIZE. */
    float w1_std = sqrtf(2.0f / (float)CNN_FLAT_SIZE);
    for (int h = 0; h < CNN_HIDDEN_SIZE; h++)
        for (int f = 0; f < CNN_FLAT_SIZE; f++)
            net->weights.W1[h][f] = randn() * w1_std;

    /* He initialisation for Dense layer 2: fan_in = CNN_HIDDEN_SIZE. */
    float w2_std = sqrtf(2.0f / (float)CNN_HIDDEN_SIZE);
    for (int k = 0; k < CNN_N_CLASSES; k++)
        for (int h = 0; h < CNN_HIDDEN_SIZE; h++)
            net->weights.W2[k][h] = randn() * w2_std;

    /* Biases are already zero from calloc. */

    return net;
}

void cnn_free(CNN *net)
{
    free(net);
}

/* -------------------------------------------------------------------------
 * Forward pass
 * ---------------------------------------------------------------------- */

/**
 * @brief Convolution stage: compute net->act.conv_out.
 *
 * For each filter f and output position (r, c):
 *   conv_out[f][r][c] = bias[f] + sum_{dr,dc} kernel[f][dr][dc] * input[r+dr][c+dc]
 *
 * @param net CNN with loaded weights and net->act.input filled.
 */
static void forward_conv(CNN *net)
{
    const CNNWeights    *w   = &net->weights;
    CNNActivations      *act = &net->act;

    for (int f = 0; f < CNN_N_FILTERS; f++) {
        for (int r = 0; r < CNN_CONV_H; r++) {
            for (int c = 0; c < CNN_CONV_W; c++) {
                float sum = w->conv_bias[f];
                for (int dr = 0; dr < CNN_KERNEL_H; dr++)
                    for (int dc = 0; dc < CNN_KERNEL_W; dc++)
                        sum += w->kernels[f][dr][dc]
                             * act->input[r + dr][c + dc];
                act->conv_out[f][r][c] = sum;
            }
        }
    }

    /* ReLU in-place over the full conv output volume. */
    relu_inplace(&act->conv_out[0][0][0],
                 CNN_N_FILTERS * CNN_CONV_H * CNN_CONV_W);
}

/**
 * @brief Max-pooling stage: compute net->act.pool_out.
 *
 * 2×2 window, stride 2.  Stores the winning row/col within each window
 * in pool_max_r / pool_max_c for use during backprop.
 *
 * @param net CNN after forward_conv().
 */
static void forward_pool(CNN *net)
{
    CNNActivations *act = &net->act;

    for (int f = 0; f < CNN_N_FILTERS; f++) {
        for (int r = 0; r < CNN_POOL_OUT_H; r++) {
            for (int c = 0; c < CNN_POOL_OUT_W; c++) {
                int   base_r = r * CNN_POOL_H;
                int   base_c = c * CNN_POOL_W;
                float best   = act->conv_out[f][base_r][base_c];
                int   best_r = 0, best_c = 0;

                for (int dr = 0; dr < CNN_POOL_H; dr++) {
                    for (int dc = 0; dc < CNN_POOL_W; dc++) {
                        float v = act->conv_out[f][base_r + dr][base_c + dc];
                        if (v > best) {
                            best   = v;
                            best_r = dr;
                            best_c = dc;
                        }
                    }
                }

                act->pool_out[f][r][c]  = best;
                act->pool_max_r[f][r][c] = best_r;
                act->pool_max_c[f][r][c] = best_c;
            }
        }
    }
}

/**
 * @brief Flatten stage: copy pool_out into act->flat, row-major.
 *
 * Index formula: flat[f * POOL_OUT_H * POOL_OUT_W + r * POOL_OUT_W + c]
 *
 * @param net CNN after forward_pool().
 */
static void forward_flatten(CNN *net)
{
    CNNActivations *act = &net->act;
    int idx = 0;

    for (int f = 0; f < CNN_N_FILTERS; f++)
        for (int r = 0; r < CNN_POOL_OUT_H; r++)
            for (int c = 0; c < CNN_POOL_OUT_W; c++)
                act->flat[idx++] = act->pool_out[f][r][c];
}

/**
 * @brief Dense layer 1 (2704 → 128) followed by ReLU.
 *
 * z1[h] = b1[h] + sum_f W1[h][f] * flat[f]
 * h1[h] = ReLU(z1[h])
 *
 * @param net CNN after forward_flatten().
 */
static void forward_dense1(CNN *net)
{
    const CNNWeights   *w   = &net->weights;
    CNNActivations     *act = &net->act;

    for (int h = 0; h < CNN_HIDDEN_SIZE; h++) {
        float sum = w->b1[h];
        for (int f = 0; f < CNN_FLAT_SIZE; f++)
            sum += w->W1[h][f] * act->flat[f];
        act->z1[h] = sum;
        act->h1[h] = sum > 0.0f ? sum : 0.0f;  /* ReLU */
    }
}

/**
 * @brief Dense layer 2 (128 → 26) followed by Softmax.
 *
 * z2[k] = b2[k] + sum_h W2[k][h] * h1[h]
 * output = softmax(z2)
 *
 * @param net CNN after forward_dense1().
 */
static void forward_dense2(CNN *net)
{
    const CNNWeights   *w   = &net->weights;
    CNNActivations     *act = &net->act;

    for (int k = 0; k < CNN_N_CLASSES; k++) {
        float sum = w->b2[k];
        for (int h = 0; h < CNN_HIDDEN_SIZE; h++)
            sum += w->W2[k][h] * act->h1[h];
        act->z2[k]     = sum;
        act->output[k] = sum;   /* will be overwritten by softmax */
    }

    softmax_inplace(act->output, CNN_N_CLASSES);
}

void cnn_forward(CNN *net, const float *image)
{
    /* Copy image into the 2-D activation buffer. */
    memcpy(&net->act.input[0][0], image,
           CNN_IMG_H * CNN_IMG_W * sizeof(float));

    forward_conv(net);
    forward_pool(net);
    forward_flatten(net);
    forward_dense1(net);
    forward_dense2(net);
}

/* -------------------------------------------------------------------------
 * Loss
 * ---------------------------------------------------------------------- */

float cnn_loss(const CNN *net, int label)
{
    float p = net->act.output[label];
    if (p < 1e-9f) p = 1e-9f;   /* clamp to avoid log(0) */
    return -logf(p);
}

/* -------------------------------------------------------------------------
 * Backward pass
 * ---------------------------------------------------------------------- */

void cnn_backward(CNN *net, int label)
{
    const CNNWeights *w   = &net->weights;
    CNNActivations   *act = &net->act;
    CNNWeights       *g   = &net->grads;

    net->batch_count++;

    /* 1. Softmax + cross-entropy gradient: dL/dz2[k] = output[k] - 1{k==label} */
    float dz2[CNN_N_CLASSES];
    for (int k = 0; k < CNN_N_CLASSES; k++)
        dz2[k] = act->output[k] - (k == label ? 1.0f : 0.0f);

    /* 2. Dense layer 2 gradients */
    float dh1[CNN_HIDDEN_SIZE];
    memset(dh1, 0, sizeof(dh1));
    for (int k = 0; k < CNN_N_CLASSES; k++) {
        g->b2[k] += dz2[k];
        for (int h = 0; h < CNN_HIDDEN_SIZE; h++) {
            g->W2[k][h] += dz2[k] * act->h1[h];
            dh1[h]      += w->W2[k][h] * dz2[k];
        }
    }

    /* 3. ReLU backprop through hidden layer */
    float dz1[CNN_HIDDEN_SIZE];
    for (int h = 0; h < CNN_HIDDEN_SIZE; h++)
        dz1[h] = dh1[h] * (act->z1[h] > 0.0f ? 1.0f : 0.0f);

    /* 4. Dense layer 1 gradients */
    float dflat[CNN_FLAT_SIZE];
    memset(dflat, 0, sizeof(dflat));
    for (int h = 0; h < CNN_HIDDEN_SIZE; h++) {
        g->b1[h] += dz1[h];
        for (int f = 0; f < CNN_FLAT_SIZE; f++) {
            g->W1[h][f] += dz1[h] * act->flat[f];
            dflat[f]    += w->W1[h][f] * dz1[h];
        }
    }

    /* 5. Unflatten dflat → dpool_out */
    float dpool_out[CNN_N_FILTERS][CNN_POOL_OUT_H][CNN_POOL_OUT_W];
    {
        int idx = 0;
        for (int f = 0; f < CNN_N_FILTERS; f++)
            for (int r = 0; r < CNN_POOL_OUT_H; r++)
                for (int c = 0; c < CNN_POOL_OUT_W; c++)
                    dpool_out[f][r][c] = dflat[idx++];
    }

    /* 6. MaxPool backprop: route gradient to the winning position only */
    float dconv_out[CNN_N_FILTERS][CNN_CONV_H][CNN_CONV_W];
    memset(dconv_out, 0, sizeof(dconv_out));
    for (int f = 0; f < CNN_N_FILTERS; f++) {
        for (int r = 0; r < CNN_POOL_OUT_H; r++) {
            for (int c = 0; c < CNN_POOL_OUT_W; c++) {
                int base_r = r * CNN_POOL_H;
                int base_c = c * CNN_POOL_W;
                int mr     = act->pool_max_r[f][r][c];
                int mc     = act->pool_max_c[f][r][c];
                dconv_out[f][base_r + mr][base_c + mc] += dpool_out[f][r][c];
            }
        }
    }

    /* 7. ReLU backprop through conv output (conv_out holds post-ReLU values) */
    for (int f = 0; f < CNN_N_FILTERS; f++)
        for (int r = 0; r < CNN_CONV_H; r++)
            for (int c = 0; c < CNN_CONV_W; c++)
                dconv_out[f][r][c] *= (act->conv_out[f][r][c] > 0.0f ? 1.0f : 0.0f);

    /* 8. Conv layer gradients */
    for (int f = 0; f < CNN_N_FILTERS; f++) {
        float db = 0.0f;
        for (int r = 0; r < CNN_CONV_H; r++) {
            for (int c = 0; c < CNN_CONV_W; c++) {
                float d = dconv_out[f][r][c];
                db += d;
                for (int dr = 0; dr < CNN_KERNEL_H; dr++)
                    for (int dc = 0; dc < CNN_KERNEL_W; dc++)
                        g->kernels[f][dr][dc] += d * act->input[r + dr][c + dc];
            }
        }
        g->conv_bias[f] += db;
    }
}

/* -------------------------------------------------------------------------
 * Weight update
 * ---------------------------------------------------------------------- */

void cnn_zero_grads(CNN *net)
{
    memset(&net->grads, 0, sizeof(CNNWeights));
}

/**
 * @brief Apply SGD+momentum to one weight array.
 *
 * For each weight w, velocity v, gradient g:
 *   v  = β * v + g
 *   w -= lr * v
 *
 * @param w      Weight array (modified in-place).
 * @param v      Velocity array (modified in-place).
 * @param g      Gradient array (read-only).
 * @param n      Number of elements.
 */
static void sgd_update(float *w, float *v, const float *g, size_t n,
                       float inv_batch)
{
    for (size_t i = 0; i < n; i++) {
        float g_avg = g[i] * inv_batch;   /* average gradient over the batch */
        v[i] = CNN_MOMENTUM * v[i] + g_avg;
        w[i] -= CNN_LR * v[i];
    }
}

void cnn_update(CNN *net)
{
    CNNWeights *w = &net->weights;
    CNNWeights *v = &net->velocity;
    CNNWeights *g = &net->grads;

    /* Compute how many samples contributed to the accumulated gradients.
     * net->batch_count is set by the training loop before calling cnn_update. */
    float inv_batch = (net->batch_count > 0)
                      ? 1.0f / (float)net->batch_count
                      : 1.0f / (float)CNN_BATCH_SIZE;

    sgd_update(&w->kernels[0][0][0], &v->kernels[0][0][0],
               &g->kernels[0][0][0],
               CNN_N_FILTERS * CNN_KERNEL_H * CNN_KERNEL_W, inv_batch);

    sgd_update(w->conv_bias, v->conv_bias, g->conv_bias,
               CNN_N_FILTERS, inv_batch);

    sgd_update(&w->W1[0][0], &v->W1[0][0], &g->W1[0][0],
               CNN_HIDDEN_SIZE * CNN_FLAT_SIZE, inv_batch);

    sgd_update(w->b1, v->b1, g->b1, CNN_HIDDEN_SIZE, inv_batch);

    sgd_update(&w->W2[0][0], &v->W2[0][0], &g->W2[0][0],
               CNN_N_CLASSES * CNN_HIDDEN_SIZE, inv_batch);

    sgd_update(w->b2, v->b2, g->b2, CNN_N_CLASSES, inv_batch);

    cnn_zero_grads(net);
    net->batch_count = 0;
}

/* -------------------------------------------------------------------------
 * Prediction
 * ---------------------------------------------------------------------- */

int cnn_predict(CNN *net, const float *image)
{
    cnn_forward(net, image);

    int   best_class = 0;
    float best_prob  = net->act.output[0];

    for (int k = 1; k < CNN_N_CLASSES; k++) {
        if (net->act.output[k] > best_prob) {
            best_prob  = net->act.output[k];
            best_class = k;
        }
    }

    return best_class;
}
