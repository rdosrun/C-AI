#include "parser.h"
#include "../data_structures/hashtable.h"

HashTable * tokenize(char * fileName){
    char ** wordList = parse(fileName);
    HashTable * hashtable = ht_create(4096);
    for(int i =0; wordList[i] !=NULL;++i){
        ht_set(hashtable,wordList[i],wordList[i]);
    }
    free(wordList);
    return hashtable;
}



