#ifndef NN_H_
#define NN_H_

#include<stdio.h>
#include<math.h>
#include<stddef.h>

#ifndef NN_MALLOC 
#include<stdlib.h>
#define NN_MALLOC malloc
#endif  //NN_MALLOC

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
/*
float d[] = {1,0,1,1} 
--> data
Mat m = {.rows = 2, .cols = 2, .es = d}
Mat m = {.rows = 4, .cols = 1, .es = d}
-->shape of data */

/*
creating a dynamic matrix 
a structure having no of rows and cols of the matrix
alongwith a point *es to point at the beginning of the matrix */

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

float rand_float(void);
float sigmoidf(float X);
typedef struct{
    size_t rows;
    size_t cols;
    //how many bytes to skip in memory to move to the next position
    size_t stride; 
    float *es;
}Mat;


float mat_sig(Mat m);
Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m, const char *name, size_t padding);        
void mat_rand(Mat m, float low, float high);
void mat_fill(Mat m, float x);
#define MAT_PRINT(m) mat_print(m, #m, 0);

typedef struct{
   size_t count; //total number of layers
   Mat *ws;//array of weights
   Mat *bs;//array of biases
   Mat *as;//amnt of activations will be count+1 since it includes input layer too
}NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);

#endif //NN_H_

























#ifdef NN_IMPLEMENTATION

float rand_float(void){
    return (float)rand() / (float)RAND_MAX;
}
float sigmoidf(float X){
    return (1.0f / (1.0f + expf(-X)));
}
float mat_sig(Mat m){//calculates sigmoid of all elements of matrix 
    for(size_t i=0; i< m.rows; i++){
        for(size_t j=0; j< m.cols ;j++){
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
    }
}
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    //moving to the next row everytime stride is called
    //it contains the total number of cols, so while calling MAT_AT(m,i,j),
    //we can skip i*stride rows and go to the jth column
    m.stride = cols;
    m.es = NN_MALLOC((sizeof(*m.es))*rows*cols);
/*sizeof(*m.es) - calculates the size of data type that m.es points to
    next time if take double instead of float, then the size of the
    data type is automatically updated and memory assigned accordingly*/
    NN_ASSERT(m.es != NULL);   
    return m;
}
void mat_dot(Mat dst, Mat a, Mat b){
    // a = 2r*3c ; b = 3r*4c
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.cols == b.cols);
    NN_ASSERT(dst.rows == a.rows);

    for(int i=0; i<dst.rows; i++){
        for(int j=0; j<dst.cols; j++){
            MAT_AT(dst, i, j) = 0;
            for(int k=0;k<n; k++){
                MAT_AT(dst, i, j) += MAT_AT(a,i,k) * MAT_AT(b,k,j);
            } 
        }
    }
}
Mat mat_row(Mat m,size_t row){
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        //stride is not needed since we just have one row here,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),//returns the address of the 0th column 
    };
}
void mat_copy(Mat dst, Mat src){
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for(size_t i = 0; i<dst.rows; i++){
        for(size_t j = 0; j<dst.cols; j++){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}
void mat_sum(Mat dst, Mat a){
    for(size_t i = 0; i< a.rows; i++){
        for(size_t j = 0; j<a.cols; j++){
            MAT_AT(dst,i,j) += MAT_AT(a,i,j);
        }
    }
}   
void mat_fill(Mat m, float x){
    for(size_t i = 0; i< m.rows; i++){
        for(size_t j = 0; j<m.cols; j++){
            MAT_AT(m,i,j) = x;
        }
    }
}
void mat_print(Mat m , const char *name, size_t padding){
    printf("%*s%s = [\n", (int) padding, "",  name);
    for(size_t i = 0; i< m.rows; i++){
        printf("%*s   ",(int) padding, "");
        for(size_t j = 0; j<m.cols; j++){
            printf("%f ", MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}
void mat_rand(Mat m, float low, float high){
    for(size_t i = 0; i< m.rows; i++){
        for(size_t j = 0; j<m.cols; j++){
            MAT_AT(m,i,j) = rand_float()*(high - low) + low;
        }
    }
}


NN nn_alloc(size_t *arch, size_t arch_count){
   NN_ASSERT(arch_count > 0);

   NN nn;
   nn.count = arch_count - 1;

   nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
   NN_ASSERT(nn.ws != NULL);
   nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
   NN_ASSERT(nn.bs != NULL);
   nn.as = NN_MALLOC(sizeof(*nn.as)*nn.count);
   NN_ASSERT(nn.as != NULL);


   nn.as[0] = mat_alloc(1, arch[0]);
   for(size_t i = 1; i < arch_count; i++){
      nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
      nn.bs[i-1] = mat_alloc(1, arch[i]);
      nn.as[i] = mat_alloc(1, arch[i]);
   }

   return nn;
}

void nn_zero(NN nn){
    for(size_t i = 0; i < nn.count; i++){
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
    
}

void nn_print(NN nn, const char *name){
    char buf[256];
    printf("%s = [\n", name);
    for(size_t i = 0; i< nn.count; i++){
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }

}

void nn_rand(NN nn, float low, float high){
    for(size_t i = 0; i< nn.count; i++){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn){
    for(size_t i = 0; i< nn.count; i++){
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);//as[1] (1,2)= as[0](1,2) * ws[1](2,1)
        mat_sum(nn.as[i+1], nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
}

float nn_cost(NN nn, Mat ti, Mat to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    
    float c = 0;
    for(size_t i = 0; i<n ; i++){
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);
        size_t q = to.cols;
        for(size_t j = 0; j<q; j++){
            float d = MAT_AT(NN_OUTPUT(nn),0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }
    return c/n;
}
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to){
    float saved;
    float c = nn_cost(nn, ti, to);
    for(size_t i = 0; i< nn.count; i++){
        for(size_t j = 0; j<nn.ws[i].rows; j++){
            for(size_t k = 0; k< nn.ws[i].cols; k++){
                saved = MAT_AT(nn.ws[i],j,k);
                MAT_AT(nn.ws[i],j,k) += eps;
                MAT_AT(g.ws[i],j,k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i],j,k) = saved;
            }
        }
        
        for(size_t j = 0; j<nn.bs[i].rows; j++){
            for(size_t k = 0; k< nn.bs[i].cols; k++){
                saved = MAT_AT(nn.bs[i],j,k);
                MAT_AT(nn.bs[i],j,k) += eps;
                MAT_AT(g.bs[i],j,k) = (nn_cost(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i],j,k) = saved;
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to){
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);
    
    nn_zero(g);

    //i - current sample
    //l - current layer
    //j - current activation
    //k - previous activation
    for(size_t i = 0; i < n; i++){
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);
        for(size_t j = 0; j <= nn.count; j++){
            mat_fill(g.as[j], 0);
        }
        
        
        for(size_t j = 0; j < to.cols; j++){
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);   
        }

        for(size_t l = nn.count; l>0; l--){
            for(size_t j = 0; j < nn.as[l].cols; j++){
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1-a);
                for(size_t k = 0; k < nn.as[l-1].cols; k++){
                    //j - weight matrix col
                    //k - weight matrix row
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1-a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }
}

// void nn_backprop(NN nn, NN g, Mat ti, Mat to){
//     NN_ASSERT(ti.rows == to.rows);
//     size_t n = ti.rows;
//     NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

//     for(size_t i = 0; i< n; i++){
//         mat_copy(NN_INPUT(nn), mat_row(ti, i));
//         nn_forward(nn);
//         for(size_t j = 0; j<to.cols; j++){
//             MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
//         }

//         for(size_t l = nn.count; l>0; l--){
//             for(size_t j = 0; j < nn.as[l].cols; j++){
                
//                 float a = MAT_AT(nn.as[l], 0, j);
//                 float da = MAT_AT(g.as[l], 0, j);
//                 MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1-a);
//                 for(size_t k = 0; k < nn.as[l-1].cols; k++){
//                     float pa = MAT_AT(nn.as[l-1], 0, k);
//                     float w = MAT_AT(nn.ws[l-1], k, j);
//                 }
//             }
//         }
//     }
// }

void nn_learn(NN nn , NN g, float rate){
    for(size_t i = 0; i< nn.count; i++){
        for(size_t j = 0; j<nn.ws[i].rows; j++){
            for(size_t k = 0; k< nn.ws[i].cols; k++){
                MAT_AT(nn.ws[i],j,k) -= rate*MAT_AT(g.ws[i],j,k);
            }
        }

        for(size_t j = 0; j<nn.bs[i].rows; j++){
            for(size_t k = 0; k< nn.bs[i].cols; k++){
                MAT_AT(nn.bs[i],j,k) -= rate*MAT_AT(g.bs[i],j,k);
            }
        }
    }
}
#endif //NN_IMPLEMENTATION
