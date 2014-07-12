//!!!!!STILL NOT FUNCTIONAL DON"T USE IT !!!!!\\

#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef struct _svm_node
{
    int index;
    double value;
}svm_node;

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

double my_dot(__global const svm_node *px, __global const svm_node *py, const int xW)
{
    double sum = 0;
    for (int i = 0; i < xW - 1; i++){
        sum += px[i].value * py[i].value;
    }
//    while(px->index != -1 && py->index != -1)
//    {
//        if(px->index == py->index)
//        {
//            sum += px->value * py->value;
//            ++px;
//            ++py;
//        }
//        else
//        {
//            if(px->index > py->index)
//                ++py;
//            else
//                ++px;
//        }
//    }
    return sum;
}

double kernel_linear(const int i, const int j, __global const svm_node* x, const int xW){
    return my_dot(&x[i * xW], &x[j * xW], xW);
}

double kernel_poly( const int i, const int j, __global const svm_node* x, const int xW, 
                    double gamma, double coef0, int degree){    
    return pow(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0, degree);
}

double kernel_rbf(  const int i, const int j, __global const svm_node* x, const int xW,
                    double gamma, __global double* x_square){   
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * my_dot(&x[i * xW], &x[j * xW], xW)));
}

double kernel_sigmoid(const int i, const int j, __global const svm_node* x, 
                      const int xW, double gamma, double coef0){
    return tanh(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0);
}

double kernel_precomputed(int i, int j, __global const svm_node* x, const int xW){
    int jIndex = (int)x[j * xW].value;
    return x[i * xW + jIndex].value;
}

__kernel void svcQgetQ( __global float* data, const int dataLen, const int start, 
                        const int len, const int i, const int kernel_type, 
                        __global const char* y, __global const svm_node* x, const int xW,
                        double gamma, double coef0, int degree, __global double* x_square)
{  
    
    const int index = get_global_id(0);
    const int realIndex = index + start;    
    if (realIndex < dataLen){
        float res = y[i] * y[realIndex];
        switch (abs(kernel_type)){
        case LINEAR:
            res *= (float)kernel_linear(i, realIndex, x, xW);
            break;
        case POLY:
            res *= (float)kernel_poly(i, realIndex, x, xW, gamma, coef0, degree);
            break;
        case RBF:
            res *= (float)kernel_rbf(i, realIndex, x, xW, gamma, x_square);
            break;
        case SIGMOID:
            res *= (float)kernel_sigmoid(i, realIndex, x, xW, gamma, coef0);
            break;
        case PRECOMPUTED:
            res *= (float)kernel_precomputed(i, realIndex, x, xW);
            break;
        }
        data[realIndex] = res;
    }
}

__kernel void svrQgetQ (__global float* data, const int dataLen, const int start, 
                        const int len, const int i, const int kernel_type,
                         __global svm_node* x, const int xW, double gamma,
                        double coef0, int degree, __global double* x_square)
{    
    const int index = get_global_id(0);
    const int realIndex = index + start;    
    if (realIndex < dataLen){
        switch (abs(kernel_type)){
        case LINEAR:
            data[realIndex] = (float)kernel_linear(i, realIndex, x, xW);
            break;
        case POLY:
            data[realIndex] = (float)kernel_poly(i, realIndex, x, xW, gamma, coef0, degree);
            break;
        case RBF:            
            data[realIndex] = (float)kernel_rbf(i, realIndex, x, xW, gamma, x_square);
            break;
        case SIGMOID:
            data[realIndex] = (float)kernel_sigmoid(i, realIndex, x, xW, gamma, coef0);
            break;
        case PRECOMPUTED:
            data[realIndex] = (float)kernel_precomputed(i, realIndex, x, xW);
            break;
        }
    }
}
