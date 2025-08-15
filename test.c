#include "stdlib.h"
#include "time.h"
#include "stdio.h"
#include "limits.h"

int main(){
    FILE * fptr =  fopen("fizzbuzz.txt","w");
    for(int i =0; i<100000;++i){
        fprintf(fptr,"%d ",i);
        if(i%3==0){
            fprintf(fptr,"fizz");
        }
        if(i%5==0){
            fprintf(fptr,"buzz");
        }
        fprintf(fptr,"\n");
    }
    return 0;
}
