// model.c — single-file training loop with one hidden layer

#include <string.h>
#include "./math/matrix_math.h"   // your matrix API (matrix_init, dot, transpose, relu, Softmax, ...)
#include <stdio.h>
#include <math.h>

#ifndef CLASS_COUNT
#define CLASS_COUNT     4      // number, fizz, buzz, fizzbuzz
#endif

#ifndef VOCAB_SIZE
#define VOCAB_SIZE      15     // mod-15 is sufficient for FizzBuzz information
#endif

#ifndef EMBEDDING_SIZE
#define EMBEDDING_SIZE  15     // D1: size after embedding
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE     128     // H: hidden layer width
#endif

#ifndef TRAINING_ITR
#define TRAINING_ITR    10      // epochs
#endif

#ifndef LR
#define LR              1e-2   // learning rate
#endif

#ifndef MAX_N
#define MAX_N           31  // train on numbers 0..MAX_N-1
#endif

// ---------- helpers (no external deps beyond your matrix ops) ----------

// Encode n as one-hot over VOCAB_SIZE (n % VOCAB_SIZE) into an existing [VOCAB_SIZE x 1] matrix.
static inline void encode_input_inplace(struct matrix *x, int n) {
    int V = x->height;
    // zero
    memset(x->grid, 0, (size_t)V * sizeof *x->grid);
    int hot = n % V;
    if (hot < 0) hot += V;
    x->grid[hot] = 1.0;
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
    // Embedding E: [VOCAB_SIZE x EMBEDDING_SIZE]
    struct matrix *E  = matrix_init(VOCAB_SIZE, EMBEDDING_SIZE);
    // Hidden layer W1,b1: [HIDDEN_SIZE x EMBEDDING_SIZE], [HIDDEN_SIZE x 1]
    struct matrix *W1 = matrix_init(HIDDEN_SIZE, EMBEDDING_SIZE);
    struct matrix *b1 = matrix_init(HIDDEN_SIZE, 1);
    // Output layer W2,b2: [CLASS_COUNT x HIDDEN_SIZE], [CLASS_COUNT x 1]
    struct matrix *W2 = matrix_init(CLASS_COUNT, HIDDEN_SIZE);
    struct matrix *b2 = matrix_init(CLASS_COUNT, 1);

    // Init
    randomize(E);
    randomize(W1);
    randomize(W2);
    // b1 and b2 default to zero (fine)

    // Work buffers (reused each step)
    struct matrix *x       = matrix_init(VOCAB_SIZE, 1);      // input one-hot
    struct matrix *y       = matrix_init(CLASS_COUNT, 1);     // target one-hot
    struct matrix *ET      = NULL;                            // E^T
    struct matrix *h_emb   = NULL;                            // [EMBEDDING_SIZE x 1]
    struct matrix *z1      = NULL;                            // pre-activation hidden [H x 1]
    struct matrix *a1      = NULL;                            // post-ReLU hidden [H x 1]
    struct matrix *logits  = NULL;                            // [C x 1], reused as probs after Softmax

    // Grad buffers (allocate new each iter or reuse; here we reuse where possible)
    struct matrix *dlogits = matrix_init(CLASS_COUNT, 1);     // [C x 1]
    struct matrix *dW2     = NULL;                            // [C x H]
    struct matrix *db2     = matrix_init(CLASS_COUNT, 1);     // [C x 1]
    struct matrix *da1     = NULL;                            // [H x 1]
    struct matrix *dW1     = NULL;                            // [H x D1]
    struct matrix *db1g    = matrix_init(HIDDEN_SIZE, 1);     // [H x 1]
    struct matrix *dh_emb  = NULL;                            // [D1 x 1]

    for (int epoch = 0; epoch < TRAINING_ITR; ++epoch) {
        double loss_sum = 0.0;
        printf("=======loop %d========\n", epoch);

        for (int n = 0; n < MAX_N; ++n) {
            // --------- Encode one sample ---------
            encode_input_inplace(x, n);         // x: [V x 1], one-hot at n%V
            target_one_hot_inplace(y, n);       // y: [C x 1]

            // --------- Forward pass ---------
            // h_emb = ReLU( E^T · x )
            if (ET) { destroy_matrix(ET);}
            ET    = transpose(E);               // [D1 x V]
            if (h_emb) {
                destroy_matrix(h_emb); }
            h_emb = dot(ET, x);                 // [D1 x 1]
            relu(h_emb);                        // in-place
            // a1 = ReLU( W1 · h_emb + b1 )
            if (z1) { destroy_matrix(z1); }
            z1 = dot(W1, h_emb);                // [H x 1]
            add_inplace(z1, b1);                // +b1
            a1 = z1;                            // alias (reuse buffer)
            relu(a1);                           // in-place

            // logits = W2 · a1 + b2
            if (logits) { destroy_matrix(logits); }
            logits = dot(W2, a1);               // [C x 1]
            add_inplace(logits, b2);            // +b2
            //print_matrix(logits);

            // probs = softmax(logits) (in-place)
            Softmax(logits);
            struct matrix *probs = logits;
            if(epoch+1 == TRAINING_ITR){
                printf("guess for %d\n",n);
                print_matrix(probs);
            }
            // loss (cross-entropy)
            double L = 0.0, eps = 1e-12;
            for (int c = 0; c < CLASS_COUNT; ++c) {
                if (y->grid[c] > 0.0) L -= log(probs->grid[c] + eps);
            }
            loss_sum += L;



            // --------- Backward pass ---------
            // dlogits = probs - y
            // (use your in-place subtract)
            // NOTE: sub_inplace modifies 'probs' buffer; safe since we won't use probs again this iter.
            sub_inplace(probs, y);              // probs := probs - y
            // Copy into dlogits (or just treat probs as dlogits)
            memcpy(dlogits->grid, probs->grid, (size_t)CLASS_COUNT * sizeof *dlogits->grid);

            // dW2 = dlogits · a1^T
            struct matrix *a1T = transpose(a1);
            if (dW2) { destroy_matrix(dW2); dW2 = NULL; }
            dW2 = dot(dlogits, a1T);            // [C x H]
            destroy_matrix(a1T);

            // db2 = dlogits
            memcpy(db2->grid, dlogits->grid, (size_t)CLASS_COUNT * sizeof *db2->grid);

            // da1 = W2^T · dlogits
            struct matrix *W2T = transpose(W2);
            if (da1) { destroy_matrix(da1); da1 = NULL; }
            da1 = dot(W2T, dlogits);            // [H x 1]
            destroy_matrix(W2T);

            // ReLU backprop at a1: da1 *= (a1 > 0)
            for (int k = 0; k < HIDDEN_SIZE; ++k) {
                if (!(a1->grid[k] > 0.0)) da1->grid[k] = 0.0;
            }

            // dW1 = da1 · h_emb^T
            struct matrix *hEmbT = transpose(h_emb);
            if (dW1) { destroy_matrix(dW1); dW1 = NULL; }
            dW1 = dot(da1, hEmbT);              // [H x D1]
            destroy_matrix(hEmbT);

            // db1 = da1
            memcpy(db1g->grid, da1->grid, (size_t)HIDDEN_SIZE * sizeof *db1g->grid);

            // dh_emb = W1^T · da1
            struct matrix *W1T = transpose(W1);
            if (dh_emb) { destroy_matrix(dh_emb); dh_emb = NULL; }
            dh_emb = dot(W1T, da1);             // [D1 x 1]
            destroy_matrix(W1T);

            // ReLU backprop at h_emb: dh_emb *= (h_emb > 0)  (we used ReLU on h_emb)
            for (int k = 0; k < EMBEDDING_SIZE; ++k) {
                if (!(h_emb->grid[k] > 0.0)) dh_emb->grid[k] = 0.0;
            }

            // --------- SGD updates ---------
            // W2 -= LR * dW2 ; b2 -= LR * db2
            {
                size_t nW2 = (size_t)W2->height * (size_t)W2->width;
                for (size_t t = 0; t < nW2; ++t) W2->grid[t] -= LR * dW2->grid[t];
                for (int c = 0; c < CLASS_COUNT; ++c) b2->grid[c] -= LR * db2->grid[c];
            }

            // W1 -= LR * dW1 ; b1 -= LR * db1
            {
                size_t nW1 = (size_t)W1->height * (size_t)W1->width;
                for (size_t t = 0; t < nW1; ++t) W1->grid[t] -= LR * dW1->grid[t];
                for (int h = 0; h < HIDDEN_SIZE; ++h) b1->grid[h] -= LR * db1g->grid[h];
            }

            // Embedding sparse update: only the row actually used in x
            int hot = find_hot_index(x);
            if (hot >= 0) {
                // E[hot, :] -= LR * dh_emb[:]
                for (int d = 0; d < EMBEDDING_SIZE; ++d)
                    E->grid[hot * EMBEDDING_SIZE + d] -= LR * dh_emb->grid[d];
            }

            // --------- cleanup per-sample temps ---------
            // a1 is alias of z1, don't free twice. We free z1 at end of iter.
            // probs/logits has been mutated; destroy at end of iter.
            //destroy_matrix(logits);   logits = NULL;
            //destroy_matrix(z1);       z1 = NULL;
            // keep h_emb and x; they are recreated every iter, so we also free here:
            //destroy_matrix(h_emb);    h_emb = NULL;
            // x and y are reused buffers allocated once; do NOT destroy here
            //printf("guess for %d\n",n);
            //print_matrix(probs);
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
    if (dW2) destroy_matrix(dW2);
    destroy_matrix(db2);
    if (da1) destroy_matrix(da1);
    if (dW1) destroy_matrix(dW1);
    destroy_matrix(db1g);
    if (dh_emb) destroy_matrix(dh_emb);

    return 0;
}

