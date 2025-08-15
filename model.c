#include "./data_structures/hashtable.h"
#include "./math/matrix_math.h"
#include "./training/tokenization.h"

#define VOCAB_SIZE 1000
#define EMBEDDING_SIZE  1000
#define TRAINING_ITR  5
#define INPUT_FILE "input.txt"


int main(int argc, char ** argv){

    //init data structures
    HashTable * hashtable;
    struct matrix * input_layer = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * hidden_layer1 = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * weight = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * output_layer = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * bias_term = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    printf("matrix created\n");

    randomize(hidden_layer1);
    //training loop

    printf("matrix randomized\n");

    hashtable = tokenize(INPUT_FILE);
    printf("matrix created\n");

    for(int i =0;i<TRAINING_ITR;++i){

    }


    //execution

        //input


    return 0;

}






