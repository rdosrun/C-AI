#include "./data_structures/hashtable.h"
#include "./math/matrix_math.h"
#include "./training/tokenization.h"

#define VOCAB_SIZE 100
#define EMBEDDING_SIZE  100
#define TRAINING_ITR  5
#define INPUT_FILE "input.txt"


int main(int argc, char ** argv){

    //init data structures
    HashTable * hashtable;
    struct matrix * input_layer = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * hidden_layer1 = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * weight = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);
    struct matrix * output_layer;
    struct matrix * bias_term = matrix_init(VOCAB_SIZE,EMBEDDING_SIZE);

    randomize(hidden_layer1);
    randomize(input_layer);
    //training loop


    hashtable = tokenize(INPUT_FILE);

    for(int i =0;i<TRAINING_ITR;++i){
        //forward prop
        output_layer = dot(hidden_layer1,input_layer);
        relu(output_layer);
        Softmax(output_layer);
    }
    print_matrix(output_layer);


    //execution

        //input


    return 0;

}






