#include "./data_structures/hashtable.h"
#include "./math/matrix_math.h"
#include "./training/tokenization.h"

#define VOCAB_SIZE 100
#define EMBEDDING_SIZE  100
#define CLASS_COUNT 4
#define TRAINING_ITR  5
#define INPUT_FILE "fizzbuzz.txt"


void fill_matrix_with_hashtable(HashTable *ht, struct matrix *m){
    for(int i =0;i<ht->capacity;++i){
        if(ht->entries[i].in_use){
            int freq = (int)ht->entries[i].count;

            int index = ((int)ht_hash_str(ht->entries[i].key))%VOCAB_SIZE;
            if(index >= 0 && index <m->height){
                m->grid[index * m->width] = (double)freq;
            }
        }
    }

}


int main(int argc, char ** argv){

    //init data structures
    HashTable * hashtable;

    struct matrix * input_layer = matrix_init(VOCAB_SIZE,1);

    struct matrix * embedding_layer = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);

    struct matrix * weight = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * bias_term = matrix_init(CLASS_COUNT,1);

    struct matrix * logits = NULL;
    struct matrix * hidden_layer1 = NULL;
    struct matrix * probs = NULL;

    struct matrix * output_layer;


    randomize(embedding_layer);
    randomize(weight);
    //training loop


    hashtable = tokenize(INPUT_FILE);
    fill_matrix_with_hashtable(hashtable,input_layer);

    for(int i =0;i<TRAINING_ITR;++i){
        //forward prop
        if (hidden_layer1){
            destroy_matrix(hidden_layer1);
        }
        hidden_layer1 = dot(transpose(embedding_layer),input_layer);
        relu(hidden_layer1);

        if(logits){
            destroy_matrix(logits);
        }
        logits = dot(weight,hidden_layer1);
        add_inplace(logits, bias_term);

        if(probs){
            destroy_matrix(probs);
        }
        Softmax(logits);
        probs = logits;

    }
    print_matrix(output_layer);


    //execution

        //input

    destroy_matrix(logits);
    destroy_matrix(input_layer);
    destroy_matrix(embedding_layer);
    destroy_matrix(weight);
    destroy_matrix(bias_term);
    destroy_matrix(hidden_layer1);
    destroy_matrix(probs);
    destroy_matrix(output_layer);



    return 0;

}






