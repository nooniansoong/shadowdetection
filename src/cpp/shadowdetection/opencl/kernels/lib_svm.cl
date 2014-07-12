//!!!!!STILL NOT FUNCTIONAL DON"T USE IT !!!!!\\

#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef struct _svm_node
{
    int index;
    double value;
}svm_node;

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

double my_dot(__global const double* px, __global const double* py, const int xW)
{
    double sum = 0;
//    for (int i = 0; i < xW - 1; i++){
//        sum += px[i] * py[i];
//    }
    int i = 0;
    while (i < xW - 1){
        int remain = xW - i;
        if (remain >= 8){
            double8 xv = vload8(i / 8, px);
            double8 yv = vload8(i / 8, py);
            double8 d = xv * yv;            
            sum += d.s0 + d.s1 + d.s2 + d.s3 + d.s4 + d.s5 + d.s6 + d.s7;
            i += 8; 
        }
        else if (remain >= 4){
            double4 xv = vload4(i / 4, px);
            double4 yv = vload4(i / 4, py);
            double4 d = xv * yv;                
            sum += d.x + d.y + d.z + d.w;
            i += 4;
        }
        else if (remain >= 2){
            double2 xv = vload2(i / 2, px);
            double2 yv = vload2(i / 2, py);
            double2 d = xv * yv;                
            sum += d.x + d.y;
            i += 2;
        }
        else if (remain >= 1){
            double d = px[i] * py[i];
            sum += d;
            i++;
        }        
    }
    return sum;
}

double kernel_linear(const int i, const int j, __global const double* x, const int xW){
    return my_dot(&x[i * xW], &x[j * xW], xW);
}

double kernel_poly( const int i, const int j, __global const double* x, const int xW, 
                    double gamma, double coef0, int degree){    
    return pow(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0, degree);
}

double kernel_rbf(  const int i, const int j, __global const double* x, const int xW,
                    double gamma, __global double* x_square){   
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * my_dot(&x[i * xW], &x[j * xW], xW)));
}

double kernel_sigmoid(const int i, const int j, __global const double* x, 
                      const int xW, double gamma, double coef0){
    return tanh(gamma * my_dot(&x[i * xW], &x[j * xW], xW) + coef0);
}

double kernel_precomputed(int i, int j, __global const double* x, const int xW){
    int jIndex = (int)x[j * xW];
    return x[i * xW + jIndex];
}

__kernel void svcQgetQ( __global float* data, const int dataLen, const int start, 
                        const int len, const int i, const int kernel_type, 
                        __global const char* y, __global const double* x, const int xW,
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
                         __global double* x, const int xW, double gamma,
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
