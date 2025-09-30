// model.c — single-file training loop with one hidden layer

#include <string.h>
#include "./math/matrix_math.h"   // your matrix API (matrix_init, dot, transpose, relu, Softmax, ...)
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef CLASS_COUNT
#define CLASS_COUNT     4      // number, fizz, buzz, fizzbuzz
#endif

#ifndef VOCAB_SIZE
#define VOCAB_SIZE      10     // binary digits used to represent inputs
#endif

#ifndef EMBEDDING_SIZE
#define EMBEDDING_SIZE  64    // D1: size after first projection
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE     128     // H: hidden layer width
#endif

#ifndef TRAINING_ITR
#define TRAINING_ITR    200      // epochs
#endif

#ifndef LR
#define LR              5e-2   // learning rate
#endif

#ifndef MAX_N
#define MAX_N           100  // train on numbers 0..MAX_N-1
#endif




// Fisher–Yates shuffle of an index array
static inline void shuffle_idx(int *a, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

// ---------- helpers (no external deps beyond your matrix ops) ----------

// Encode n in binary into an existing [VOCAB_SIZE x 1] matrix (most significant bit first).
static inline void encode_input_inplace(struct matrix *x, int n) {
    int bits = x->height;
    memset(x->grid, 0, (size_t)bits * sizeof *x->grid);
    unsigned int value = (unsigned int)(n < 0 ? -n : n);
    for (int i = 0; i < bits && value > 0; ++i) {
        int idx = bits - 1 - i;
        x->grid[idx] = (double)(value & 1u);
        value >>= 1u;
    }
}

// One-hot target [CLASS_COUNT x 1] per FizzBuzz rules.
static inline void target_one_hot_inplace(struct matrix *y, int n) {
    memset(y->grid, 0, (size_t)CLASS_COUNT * sizeof *y->grid);
    int idx;
    if (n % 15 == 0)      idx = 3;   // fizzbuzz
    else if (n % 3 == 0)  idx = 1;   // fizz
    else if (n % 5 == 0)  idx = 2;   // buzz
    else                  idx = 0;   // number
    y->grid[idx] = 1.0;
}

// Find hot index from a one-hot input vector [VOCAB_SIZE x 1].
static inline int find_hot_index(const struct matrix *x) {
    for (int r = 0; r < x->height; ++r) {
        if (x->grid[r] > 0.5) return r;
    }
    return -1;
}

// ---------- main training ----------

int main(void) {
    printf("Starting training\n");

    // Parameters
    struct matrix *E  = matrix_init(EMBEDDING_SIZE, VOCAB_SIZE);
    struct matrix *W1 = matrix_init(HIDDEN_SIZE, EMBEDDING_SIZE);
    struct matrix *b1 = matrix_init(HIDDEN_SIZE, 1);
    struct matrix *W2 = matrix_init(CLASS_COUNT, HIDDEN_SIZE);
    struct matrix *b2 = matrix_init(CLASS_COUNT, 1);

    // Init
    randomize(E);
    randomize(W1);
    randomize(W2);

    // Work buffers (reused each step)
    struct matrix *x       = matrix_init(VOCAB_SIZE, 1);      // input bits
    struct matrix *y       = matrix_init(CLASS_COUNT, 1);     // target one-hot

    // Grad buffers
    struct matrix *dlogits = matrix_init(CLASS_COUNT, 1);
    struct matrix *db2     = matrix_init(CLASS_COUNT, 1);
    struct matrix *db1g    = matrix_init(HIDDEN_SIZE, 1);

    for (int epoch = 0; epoch < TRAINING_ITR; ++epoch) {
        double loss_sum = 0.0;
        printf("=======loop %d========\n", epoch);
       int idx[MAX_N];
        for (int i = 0; i < MAX_N; ++i) {
            idx[i] = i;
        }
        shuffle_idx(idx, MAX_N);
        for (int t = 0; t < MAX_N; ++t) {
            // --------- Encode one sample ---------
            //int n = rand()%(MAX_N+1);//idx[t];
            int n = idx[t];
            encode_input_inplace(x, n);
            target_one_hot_inplace(y, n);

            // --------- Forward pass ---------
            struct matrix *z_emb = dot(E, x);        // [D1 x 1]
            struct matrix *h_emb = matrix_copy(z_emb);
            relu(h_emb);                             // in-place

            struct matrix *a1 = dot(W1, h_emb);      // [H x 1]
            add_inplace(a1, b1);                     // +b1
            struct matrix *z1 = matrix_copy(a1);
            relu(a1);                                // in-place

            struct matrix *logits = dot(W2, a1);     // [C x 1]
            add_inplace(logits, b2);                 // +b2

            Softmax(logits);
            struct matrix *probs = logits;
            if (epoch + 1 == TRAINING_ITR) {
                printf("guess for %d\n", n);
                print_matrix(probs);
            }

            // loss (cross-entropy)
            double L = 0.0, eps = 1e-9;
            for (int c = 0; c < CLASS_COUNT; ++c) {
                if (y->grid[c] > 0.0) L -= log(probs->grid[c] + eps);
            }
            loss_sum += L;

            // --------- Backward pass ---------
            // dlogits = probs - y
            memcpy(dlogits->grid, probs->grid, (size_t)CLASS_COUNT * sizeof *dlogits->grid);
            sub_inplace(dlogits, y);

            // dW2 = dlogits · a1^T
            struct matrix *a1T = transpose(a1);
            struct matrix *dW2 = dot(dlogits, a1T);
            destroy_matrix(a1T);

            // db2 = dlogits
            memcpy(db2->grid, dlogits->grid, (size_t)CLASS_COUNT * sizeof *db2->grid);

            // da1 = W2^T · dlogits
            struct matrix *W2T = transpose(W2);
            struct matrix *da1 = dot(W2T, dlogits);
            destroy_matrix(W2T);

            // ReLU backprop at a1: da1 *= (z1 > 0)
            for (int k = 0; k < HIDDEN_SIZE; ++k) {
                if (!(z1->grid[k] > 0.0)) da1->grid[k] = 0.0;
            }

            // dW1 = da1 · h_emb^T
            struct matrix *hEmbT = transpose(h_emb);
            struct matrix *dW1 = dot(da1, hEmbT);
            destroy_matrix(hEmbT);

            // db1 = da1
            memcpy(db1g->grid, da1->grid, (size_t)HIDDEN_SIZE * sizeof *db1g->grid);

            // dh_emb = W1^T · da1
            struct matrix *W1T = transpose(W1);
            struct matrix *dh_emb = dot(W1T, da1);
            destroy_matrix(W1T);

            // ReLU backprop at h_emb
            for (int k = 0; k < EMBEDDING_SIZE; ++k) {
                if (!(z_emb->grid[k] > 0.0)) dh_emb->grid[k] = 0.0;
            }

            // dE = dh_emb · x^T
            struct matrix *xT = transpose(x);
            struct matrix *dE = dot(dh_emb, xT);
            destroy_matrix(xT);

            // --------- SGD updates ---------
            // W2 -= LR * dW2 ; b2 -= LR * db2
            size_t nW2 = (size_t)W2->height * (size_t)W2->width;
            for (size_t t = 0; t < nW2; ++t) W2->grid[t] -= LR * dW2->grid[t];
            for (int c = 0; c < CLASS_COUNT; ++c) b2->grid[c] -= LR * db2->grid[c];

            // W1 -= LR * dW1 ; b1 -= LR * db1
            size_t nW1 = (size_t)W1->height * (size_t)W1->width;
            for (size_t t = 0; t < nW1; ++t) W1->grid[t] -= LR * dW1->grid[t];
            for (int h = 0; h < HIDDEN_SIZE; ++h) b1->grid[h] -= LR * db1g->grid[h];

            // E -= LR * dE
            size_t nE = (size_t)E->height * (size_t)E->width;
            for (size_t t = 0; t < nE; ++t) E->grid[t] -= LR * dE->grid[t];

            // Clean up temporary matrices
            destroy_matrix(z_emb);
            destroy_matrix(h_emb);
            destroy_matrix(a1);
            destroy_matrix(z1);
            destroy_matrix(logits);
            destroy_matrix(dW2);
            destroy_matrix(da1);
            destroy_matrix(dW1);
            destroy_matrix(dh_emb);
            destroy_matrix(dE);
        }
        printf("avg loss: %.6f\n", loss_sum / (double)MAX_N);
    }

    // Cleanup
    destroy_matrix(E);
    destroy_matrix(W1); destroy_matrix(b1);
    destroy_matrix(W2); destroy_matrix(b2);
    destroy_matrix(x);
    destroy_matrix(y);
    destroy_matrix(dlogits);
    destroy_matrix(db2);
    destroy_matrix(db1g);

    return 0;
}
