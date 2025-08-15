#include "stdlib.h"
#include "time.h"
#include "stdio.h"

int main(){
    srand(time(0));
    printf("%f",(float)rand()/RAND_MAX);
    return 0;
}
