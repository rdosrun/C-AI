#include "../data_structures/hashtable.h"
#include "string.h"

#define MAX_LIST_SIZE 1000000


char ** parse(char * filename){
    FILE *fptr = fopen(filename,"r");
    char buffer[1024];
    char ** word_list = malloc(MAX_LIST_SIZE * sizeof(char *));
    char * tmp = malloc(sizeof(char)*(1024+1));
    int tmp_count =0;
    int line_count =0;
    while(fgets(buffer,sizeof(buffer),fptr)){
        for(int i =0;i<1024&&buffer[i]!='\0';++i){
            if((buffer[i] =='\n') && tmp_count>0){
                word_list[line_count] = malloc(sizeof(char)*(tmp_count+1));
                memcpy(word_list[line_count],tmp,tmp_count*sizeof(char));
                memset(tmp,'\0',sizeof(char)*(1024));
                ++line_count;
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





