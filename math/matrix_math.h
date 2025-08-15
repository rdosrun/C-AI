#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

//create a matrix for each color channel
struct matrix{
    int width;
    int height;
    double * grid;
};


//function forward declaration
struct matrix * dot(struct matrix * m1, struct matrix * m2);
struct matrix * matrix_init(int height,int width);
void print_matrix(struct matrix *m);
void relu(struct matrix *m);
void Sigmoid(struct matrix *m);
void Softmax(struct matrix *m);
void randomize(struct matrix *m);


void randomize(struct matrix *m){
    srand(time(NULL));
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = (double)rand()/RAND_MAX;
    }
}

void relu(struct matrix *m){
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = m->grid[i] > 0 ? m->grid[i]:0;
    }
}

void Sigmoid(struct matrix *m){
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = 1/(1+exp(m->grid[i]));
    }
}

void Softmax(struct matrix *m){
    double sum = 0;
    for(int i =0; i<m->width*m->height;++i){
        sum = sum + exp(m->grid[i]);
    }
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = exp(m->grid[i])/sum;
    }
}

void print_matrix(struct matrix *m){
    for(int i =0;i<m->width*m->height;++i){
        printf(" %f ",m->grid[i]);
        if((i+1)%m->width==0){
            printf("\n");
        }
    }

}


struct matrix * matrix_init(int height, int width){
    struct matrix * m = malloc(sizeof(struct matrix));
    m->height = height;
    m->width = width;
    m->grid = malloc(sizeof(double)* width *height);
    return m;
}


struct matrix * dot(struct matrix * m1, struct matrix * m2){
    //init values to stack
    int height1 = m1->height;
    int width1 = m1 ->width;
    int height2 = m2->height;
    int width2 = m2->width;

    struct matrix * m3 = matrix_init(height1,width2);

    //processing
    double tmp =0;
    int i =0;
    int j =0;
    for(i =0; i<height1*width2;++i){
        int m1_offset = i/height1*width1;
        int m2_offset = i%width2;
        tmp = 0;
        for(j =0; j<height2;++j){
            tmp = tmp + (m1->grid[j+m1_offset]*m2->grid[(j*width2)+m2_offset]);
        }
        m3->grid[i] = tmp;
        tmp =0;
    }
    return m3;
}






