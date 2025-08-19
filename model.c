#include "./data_structures/hashtable.h"
#include "./math/matrix_math.h"
#include "./training/tokenization.h"

#define VOCAB_SIZE 32
#define EMBEDDING_SIZE  1
#define CLASS_COUNT 4
#define TRAINING_ITR  500
#define INPUT_FILE "fizzbuzz.txt"
#define LR .01


void fill_matrix_with_hashtable(HashTable *ht, struct matrix *m){
    int index = 0;
    for(int i =0;i<ht->capacity;++i){
        if(ht->entries[i].count){
            int freq = (int)ht->entries[i].count;
            if(freq>1){
                printf("%s, %d \n",ht->entries[i].value,ht->entries[i].count);
            }
            //int index = //((int)ht_hash_str(ht->entries[i].key))%VOCAB_SIZE;
            if(index >= 0 && index <m->height){
                m->grid[index] = (double)freq;
                ++index;
            }
        }
    }

}

struct matrix * generate_answer(int input){
    struct matrix * m = matrix_init(CLASS_COUNT,1);
    if(input%15==0){
        m->grid[3] = 1;
        return m;
    }
    if(input%3==0){
        m->grid[1] = 1;
        return m;
    }
    if(input%5==0){
        m->grid[2] = 1;
        return m;
    }
    m->grid[0] = 1;
    return m;
}




int main(int argc, char ** argv){

    //init data structures
    HashTable * hashtable;

    struct matrix * input_layer = matrix_init(VOCAB_SIZE,1);

    struct matrix * embedding_layer = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);

    struct matrix * weight = matrix_init(CLASS_COUNT,EMBEDDING_SIZE);
    struct matrix * bias_term = matrix_init(CLASS_COUNT,1);

    struct matrix * logits = NULL;
    struct matrix * hidden_layer1 = NULL;
    struct matrix * probs = NULL;

    struct matrix * output_layer;


    struct matrix * dlogits =NULL;
    struct matrix * dW = NULL;
    struct matrix * db = NULL;
    struct matrix * dh = NULL;

    struct matrix * dh_pre = NULL;


    randomize(embedding_layer);
    randomize(weight);
    //training loop


    hashtable = tokenize(INPUT_FILE);
    //fill_matrix_with_hashtable(hashtable,input_layer);
    //print_matrix(input_layer);
    printf("Starting training\n");
    for(int i =0;i<TRAINING_ITR;++i){
        //data set loop
        //
        double loss_sum = 0.0;
        printf("=======loop %d========\n",i);
        for(int j =0;j<10000;++j){
               //forward prop
            //printf("  ====forward prop====\n");
            input_layer = encode_input(j,32);
            if (hidden_layer1){
                destroy_matrix(hidden_layer1);
                hidden_layer1 = NULL;
            }
            //printf("transpose\n");
            hidden_layer1 = dot(transpose(embedding_layer),input_layer);
            relu(hidden_layer1);
            //printf("completed relu\n");
            if(logits){
                destroy_matrix(logits);
                logits = NULL;
            }
            //printf("dot\n");
            logits = dot(weight,hidden_layer1);
            if(probs){
                destroy_matrix(probs);
                probs = NULL;
            }
            probs = add_inplace(logits, bias_term);
            //printf("completed bias term\n");

            Softmax(probs);
            //printf("completed Softmax \n");
            printf("guess for %d\n",j);
            print_matrix(probs);

            struct matrix * one_hot = generate_answer(j);

            double eps = 1e-12, L = 0.0;
            for (int c = 0; c < CLASS_COUNT; ++c)
                if (one_hot->grid[c] > 0.0) L -= log(logits->grid[c] + eps);
            loss_sum += L;

            //back prop
            //printf("  ====back prop====\n");
            dlogits = sub_inplace(probs,one_hot);
            dW = dot(dlogits,transpose(hidden_layer1));
            dh = dot(transpose(weight),dlogits);
            relu(dh);
            //printf("====== completed back prop=========\n");


            for (size_t i = 0, nW = (size_t)weight->height * weight->width; i < nW; ++i)
                weight->grid[i] -= LR  * dW->grid[i];


            //printf("====== updated weight=========\n");
            // b -= LR * db
            for (int i = 0; i < CLASS_COUNT; ++i)
                bias_term->grid[i] -= LR * dlogits->grid[i];


        }
        printf("avg loss: %.6f\n",  loss_sum / 10000);


    }

    printf("Completed training \n");

    print_matrix(logits);


    //execution

        //input

    destroy_matrix(logits);
    destroy_matrix(input_layer);
    destroy_matrix(embedding_layer);
    destroy_matrix(weight);
    printf("freeing bias_term\n");
    destroy_matrix(bias_term);
    destroy_matrix(hidden_layer1);
    //destroy_matrix(output_layer);



    return 0;

}






