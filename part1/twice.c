#include<stdio.h>
#include<stdlib.h>
#include<time.h>

float train[][2] = {
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},
};

//total number of elements in train (total number of pairs)
#define train_count (sizeof(train)/sizeof(train[0]))

//Generated a random float number between 0 and 1
float rand_float(void){
    return (float) rand()/ (float) RAND_MAX;
}

//cost function 
/*
1. take difference and square
2. sum the difference
3. divide by total training ex:
*/
//by using finite difference method
/*float cost(float w, float b){
    float result = 0.0f;
    for( size_t i = 0; i<train_count; i++){
        float x = train[i][0];
        float y = x*w + b;
        float d = y - train[i][1];
        result += d*d;  //square of sum of difference between predicted and actual values 
    }
    result /= train_count;
    return result;
}*/


//BY ACTUAL DERIVATIVE:
float cost(float w){
    float result = 0.0f;
    for( size_t i = 0; i<train_count; i++){
        float x = train[i][0];
        float y = x*w;
        float d = y - train[i][1];
        result += d*d;  
    }
    result /= train_count;
    return result;
}

//by using actual derivative(gradient descent)  
float dcost(float w){
    float result = 0.0f;
    size_t n = train_count;
    for(size_t i = 0; i< n ; i++){
        float x = train[i][0];
        float y = train[i][1];
        result += 2*(x*w-y)*x;
    }
    result /= n;
    return result;
}

int main(){
    srand(79);
    float w = rand_float()*10.0f; //random no. b/w 0 and 10
    float b = rand_float()*5.0f; 
    
    float eps = 1e-3;
    float rate = 1e-1;

    //BY FINITE DIFFERENCE:
    /*for(size_t i = 0; i<1000 ; i++){
        float c =  cost(w, b);
        float dw = (cost(w + eps, b) - c) /eps; // this is the derivative of the funciton
        float db = (cost(w, b + eps) - c) /eps; 
        w -= rate*dw; //subtracting the derivative from the weight using a learning rate
        b -= rate*db;
        printf("Cost = %f , w = %f, b = %f\n", c, w, b);
    }*/

    //BY ACTUAL DERIVATIVE:
    for(size_t i = 0; i<10 ; i++){
        float dw = dcost(w);
        w -= rate*dw; //subtracting the derivative from the weight using a learning rate
        printf("Cost = %f , w = %f\n",cost(w), w);
    }
    
    printf("-----------------\n");
    // printf("w = %f, b = %f\n",w, b);
    printf("w = %f\n",w);
    return 0;
}

