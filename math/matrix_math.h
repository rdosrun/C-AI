#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include "float.h"

//create a matrix for each color channel
struct matrix{
    long width;
    long height;
    double * grid;
};


//function forward declaration
struct matrix * dot(struct matrix * m1, struct matrix * m2);
struct matrix * matrix_init(int height,int width);
void print_matrix(struct matrix *m);
void relu(struct matrix *m);
void inverse_relu(struct matrix *m);
void Sigmoid(struct matrix *m);
void Softmax(struct matrix *m);
void randomize(struct matrix *m);
void destroy_matrix(struct matrix *m);
struct matrix * transpose(struct matrix *m);
void add_inplace(struct matrix *a, struct matrix *m);
struct matrix *  matrix_copy(struct matrix *m);
struct matrix * encode_input(int n, int max_bits);
void sub_inplace(struct matrix *a, struct matrix *m);
struct matrix * mult_const(struct matrix *m, double a);
double matrix_sum(struct matrix *m);



double matrix_sum(struct matrix *m){
    double total = 0.0;
    for(int i = 0;i<m->height*m->width;++i){
        total += m->grid[i];
    }
    return total;

}


struct matrix *  mult_const(struct matrix *m, double a){
    struct matrix * b = matrix_init(m->height,m->width);
    for(int i = 0;i<m->width*m->height;++i){
        b->grid[i] *= m->grid[i] *a;
    }
    return b;
}


void  sub_inplace(struct matrix *a, struct matrix *m){
    if(a->width != m->width || a->height != m->height){
        printf("add_inplace: shape mismatch \n");
        return ;
    }

    for(int i =0;i<m->width*m->height;++i){
        a->grid[i] -= m->grid[i];
    }

}

struct matrix * encode_input(int n, int max_bits){
    struct matrix * m = matrix_init(max_bits,1);

    // zero out everything
    for (int i = 0; i < max_bits; i++) {
        m->grid[i] = 0.0;
    }

    // fill from most significant bit down
    for (int i = max_bits - 1; i >= 0 && n > 0; i--) {
        m->grid[i] = n % 2;
        n /= 2;
    }

    return m;
}

/*struct matrix * encode_input(int n, int max_bits){
    struct matrix * m = matrix_init(max_bits,1);
    int i =0;
    while(n>0){
        m->grid[i++] = n%2;
        n /= 2;
    }
    return m;
}*/


struct matrix * matrix_copy(struct matrix *m){
    struct matrix * a = matrix_init(m->height,m->width);
    memcpy(a->grid,m->grid,sizeof(double) * (m->height) * (m->width));
    return a;
}


void add_inplace(struct matrix *a, struct matrix *m){
    if(a->width != m->width || a->height != m->height){
        printf("width %d , height %d | width %d ,height %d\n", a->width,a->height,m->width,m->height);
        printf("add_inplace: shape mismatch \n");
        return ;
    }

    for(int i =0;i<m->width*m->height;++i){
        a->grid[i] += m->grid[i];
    }
}


struct matrix * transpose(struct matrix *m){
    struct matrix * tmp = matrix_init(m->width,m->height);
    for(int i =0;i<m->height;++i){
        for(int j=0;j<m->width;++j){
            tmp->grid[j*tmp->width + i] = m->grid[i*m->width +j];
        }
    }
    return tmp;
}


void destroy_matrix(struct matrix *m){
    if(!m) return;
    if(m->grid){
        free(m->grid);
        m->grid = NULL;
    }
    free(m);
}

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


void inverse_relu(struct matrix *m){
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = m->grid[i] > 0 ? 1:0;
    }
}



void Sigmoid(struct matrix *m){
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = 1/(1+exp(m->grid[i]));
    }
}

// Example of a numerically stable Softmax
void Softmax(struct matrix *m) {
    // Find the maximum value in the matrix (logits)
    int length = m->height * m->width;
    for(int j =0;j<length;j=j+m->height){
        double max_val = 0.0;
        for (int i = 1; i < m->width; i++) {
            if (m->grid[i+j] > max_val) {
                max_val = m->grid[i+j];
            }
        }

        // Subtract max for stability, then exponentiate and sum
        double sum = 0.0;
        for (int i = 0; i < m->width; i++) {
            m->grid[i+j] = exp(m->grid[i+j] - max_val);
            sum += m->grid[i+j];
        }

        // Normalize
        for (int i = 0; i < m->width; i++) {
            m->grid[i+j] /= sum;
        }
    }
}

/*void Softmax(struct matrix *m) {
    const int N = m->height * m->width;
    if (N <= 0) return;
    double maxv = -INFINITY;
    for (int i=0;i<N;++i) if (m->grid[i] > maxv) maxv = m->grid[i];
    double sum = 0.0;
    for (int i=0;i<N;++i) { m->grid[i] = exp(m->grid[i] - maxv); sum += m->grid[i]; }
    double inv = 1.0 / (sum > 0.0 ? sum : 1e-300); // avoid /0 but no uniform jump
    for (int i=0;i<N;++i) m->grid[i] *= inv;
}
*/
/*void Softmax(struct matrix *m){
    double sum = 0;
    double max = 0;
    for(int i =0; i<m->width*m->height;++i){
        if(m->grid[i]>max){
            max = m->grid[i];
        }
    }
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = exp(m->grid[i]-max);
        sum = sum + m->grid[i];
    }
    for(int i =0; i<m->width*m->height;++i){
        m->grid[i] = m->grid[i]/sum;
    }
}*/

void print_matrix(struct matrix *m){
    for(int i =0;i<m->width*m->height;++i){
        printf("%d %f ",i,m->grid[i]);
        if((i+1)%m->width==0){
            printf("\n");
        }
    }

}


struct matrix * matrix_init(int height, int width){
    struct matrix * m = malloc(sizeof(struct matrix ));
    m->height = height;
    m->width = width;
    m->grid = malloc(sizeof(double)* width *height);
    for(int i =0;i<height*width;++i){
        m->grid[i] = 0;
    }
    return m;
}




struct matrix * dot(struct matrix * m1, struct matrix * m2){
    //init values to stack
    int height1 = m1->height;
    int width1 = m1 ->width;
    int height2 = m2->height;
    int width2 = m2->width;

    if (width1 != height2) {
        fprintf(stderr, "dot: shape mismatch (%dx%d) · (%dx%d)\n",
                height1, width1, height2, width2);
        return NULL;
    }


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






