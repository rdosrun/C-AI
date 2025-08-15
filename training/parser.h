#include "../data_structures/hashtable.h"
#include "string.h"

#define MAX_LIST_SIZE 100000


char ** parse(char * filename){
    FILE *fptr = fopen(filename,"r");
    char buffer[1024];
    char ** word_list = malloc(MAX_LIST_SIZE * sizeof (*word_list));
    char * tmp = malloc(1024+1);
    int tmp_count =0;
    while(fgets(buffer,sizeof(buffer),fptr)){
        for(int i =0;i<sizeof(buffer);++i){
            if(buffer[i] ==' ' || buffer[i] =='\n'){
                word_list[i] = tmp;
                tmp[0] = '\0';
                tmp_count=0;
            }
            else{
                tmp[tmp_count] = buffer[i];
                ++tmp_count;
            }
        }
    }
    fclose(fptr);
    return word_list;
}





