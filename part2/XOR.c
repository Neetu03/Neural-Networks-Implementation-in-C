#include<time.h>
#define NN_IMPLEMENTATION
# include "nn.h"
// #define NN_MALLOC my_malloc
#include<stdio.h>

typedef struct{
    Mat a0;
    Mat w1,b1,a1;
    Mat w2,b2,a2;
}Xor;

Xor xor_alloc(){
    Xor m;
    m.a0 = mat_alloc(1,2);    
    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);
    m.a1 = mat_alloc(1,2);
    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a2 = mat_alloc(1,1);
    return m;
}

void forward_xor(Xor m){
    mat_dot(m.a1, m.a0, m.w1); //taking the first layer and multiplying it by weights
    mat_sum(m.a1,m.b1);//adding the biases to the multiplication
    mat_sig(m.a1);

    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}

/*
- we are taking the input matrix
- iterating row by row
- copying each row into the a0 matrix
- calculating the xor for that row and storing it as y
- calculating the squared diff for each row and summing it up in the cost
- calculating the mean cost*/
float cost(Xor m, Mat ti, Mat to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);

    float c = 0;
    size_t n = ti.rows;
    for(size_t i = 0; i<n; i++){
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
    
        mat_copy(m.a0, x);
        forward_xor(m);
        //m.a2 me h wo output jo calculate hua h
        //y ke andar h wo output jo actual me aana h
        //therefore, dono ki no. of columns equal honi chie
        size_t q = to.cols;

        for(size_t j = 0; j < q; j++){
            float d = MAT_AT(m.a2,0,j) - MAT_AT(y,0,j);
            c += d*d;
        }   
    }
    return c/n;
}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to){
    float saved;
    float c = cost(m, ti, to);

    for(size_t i = 0; i<m.w1.rows; i++){
        for(size_t j = 0; j<m.w1.cols; j++){
            saved = MAT_AT(m.w1,i,j);
            MAT_AT(m.w1,i,j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m , ti, to) - c)/eps;
            MAT_AT(m.w1,i,j) = saved;
        }
    }
    for(size_t i = 0; i<m.b1.rows; i++){
        for(size_t j = 0; j<m.b1.cols; j++){
            saved = MAT_AT(m.b1,i,j);
            MAT_AT(m.b1,i,j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m , ti, to) - c)/eps;
            MAT_AT(m.b1,i,j) = saved;
        }
    }
    for(size_t i = 0; i<m.w2.rows; i++){
        for(size_t j = 0; j<m.w2.cols; j++){
            saved = MAT_AT(m.w2,i,j);
            MAT_AT(m.w2,i,j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m , ti, to) - c)/eps;
            MAT_AT(m.w2,i,j) = saved;
        }
    }  
    for(size_t i = 0; i<m.b2.rows; i++){
        for(size_t j = 0; j<m.b2.cols; j++){
            saved = MAT_AT(m.b2,i,j);
            MAT_AT(m.b2,i,j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m , ti, to) - c)/eps;
            MAT_AT(m.b2,i,j) = saved;
        }
    } 
}

void xor_learn(Xor m, Xor g, float rate){
    for(size_t i = 0; i<m.w1.rows; i++){
        for(size_t j = 0; j<m.w1.cols; j++){
            MAT_AT(m.w1, i, j) -= rate * MAT_AT(g.w1, i, j);
        }
    }
    for(size_t i = 0; i<m.b1.rows; i++){
        for(size_t j = 0; j<m.b1.cols; j++){
            MAT_AT(m.b1, i, j) -= rate * MAT_AT(g.b1, i, j);
        }
    }
    for(size_t i = 0; i<m.w2.rows; i++){
        for(size_t j = 0; j<m.w2.cols; j++){
            MAT_AT(m.w2, i, j) -= rate * MAT_AT(g.w2, i, j);
        }
    }  
    for(size_t i = 0; i<m.b2.rows; i++){
        for(size_t j = 0; j<m.b2.cols; j++){
           MAT_AT(m.b2, i, j) -= rate * MAT_AT(g.b2, i, j);
        }
    }
}

float td_xor[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float td_or[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main(void){
    srand(0);

    float *td = td_xor;
    size_t stride = 3;
    //sizeof(td) = 12*4 bytes; sizeof(td[0]) = 1*4bytes(just one element); 
    //n = total number of rows in td
    size_t n = 4;
    
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td//starting point 
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    float eps = 1e-1;
    float rate = 1e-1;
    
    // size_t arch[] = {2,2,2,1};
    // NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    // NN g = nn_alloc(arch, ARRAY_LEN(arch));
    // nn_rand(nn, 0 ,1);

    // NN_PRINT(nn);
    // return 0;

    //NORMAL METHOD:
    // printf("cost = %f\n", nn_cost(nn, ti, to));
    // for(size_t i =0; i<20000; i++){
    //     nn_finite_diff(nn, g, eps, ti, to);
    //     nn_learn(nn, g, rate);
    //     printf("%zu: cost = %f\n",i, nn_cost(nn, ti, to));

    // }

    //BACKPROPAGATION:
    // printf("cost = %f\n", nn_cost(nn, ti, to));
    // for(size_t i =0; i<1000; i++){
    //     nn_backprop(nn, g, ti, to);
    //     nn_learn(nn, g, rate);
    //     printf("%zu: cost = %f\n",i, nn_cost(nn, ti, to));

    // }
    


    // printf("---------------------------\n");
    // for(size_t i = 0; i<2; i++){
    //     for(size_t j = 0; j<2; j++){
    //         MAT_AT(NN_INPUT(nn),0,0) = i;
    //         MAT_AT(NN_INPUT(nn),0,1) = j;
    //         nn_forward(nn);
    //         printf("%zu ^ %zu = %f \n",i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
    //     }
    // }

    // return 0;
    // Mat row = mat_row(ti, 1);
    // MAT_PRINT(row);
    // mat_copy(NN_INPUT(nn), row);
    // nn_forward(nn);
    // MAT_PRINT(NN_OUTPUT(nn));

    


    Xor m = xor_alloc();
    Xor g = xor_alloc();
    
    mat_rand(m.w1,0,1);
    mat_rand(m.b1,0,1);
    mat_rand(m.w2,0,1);
    mat_rand(m.b2,0,1);


    

    printf("COST: %f\n", cost(m,ti,to));
    for(size_t i = 0; i<1000*10; i++){
        finite_diff(m ,g , eps, ti, to);
        xor_learn(m, g, eps);
        printf("%zu: COST: %f\n",i, cost(m,ti,to));
    }

    printf("---------------------------\n");
    for(size_t i = 0; i<2; i++){
        for(size_t j = 0; j<2; j++){
            MAT_AT(m.a0,0,0) = i;
            MAT_AT(m.a0,0,1) = j;
            forward_xor(m);
            float y = *m.a2.es; 
            printf("%zu ^ %zu = %f \n",i, j, y);
        }
    }
    return 0;
}    