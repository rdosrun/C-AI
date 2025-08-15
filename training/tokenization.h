#include "parser.h"
#include "../data_structures/hashtable.h"

HashTable * tokenize(char * fileName){
    printf("word List start\n");
    char ** wordList = parse(fileName);
    printf("parsed\n");
    HashTable * hashtable = ht_create(4096);
    printf("created hash table\n");
    for(int i =0; wordList[i] !=NULL;++i){
        ht_set(hashtable,wordList[i],wordList[i]);
    }
    free(wordList);
    return hashtable;
}



