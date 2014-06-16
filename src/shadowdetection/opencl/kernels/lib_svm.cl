//!!!!!STILL NOT FUNCTIONAL DON"T USE IT !!!!!\\

#pragma OPENCL EXTENSION cl_khr_fp64: enable

typedef struct _svm_node
{
    int index;
    double value;
}svm_node;

typedef struct _svm_parameter
{
    int svm_type;
    int kernel_type;
    int degree;	/* for poly */
    double gamma;	/* for poly/rbf/sigmoid */
    double coef0;	/* for poly/sigmoid */

    /* these are for training only */
    double cache_size; /* in MB */
    double eps;	/* stopping criteria */
    double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
    int nr_weight;		/* for C_SVC */
    int weight_label[1];	/* for C_SVC */
    double weight[1];		/* for C_SVC */
    double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;	/* for EPSILON_SVR */
    int shrinking;	/* use the shrinking heuristics */
    int probability; /* do probability estimates */
}svm_parameter;

#define MAX_NUM_OF_NODE_RECORDS 1000000
#define MAX_NUM_OF_NODES_IN_RECORD 64
#define MAX_NUM_COEFS 2016 //64 * 63 /2

//typedef struct _svm_model
//{
//    struct svm_parameter param;	/* parameter */
//    int nr_class;		/* number of classes, = 2 in regression/one class svm */
//    int l;			/* total #SV */
//    struct svm_node SV[MAX_NUM_OF_NODE_RECORDS][MAX_NUM_OF_NODES_IN_RECORD];    /* SVs (SV[l]) */
//    double sv_coef[MAX_NUM_OF_NODES_IN_RECORD][MAX_NUM_OF_NODE_RECORDS][;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
//    double rho[MAX_NUM_COEFS];                  /* constants in decision functions (rho[k*(k-1)/2]) */
//    double probA[MAX_NUM_COEFS];		/* pariwise probability information */
//    double probB[MAX_NUM_COEFS];
//    int sv_indices[MAX_NUM_OF_NODE_RECORDS];        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */
//
//    /* for classification only */
//
//    int label[MAX_NUM_OF_NODES_IN_RECORD];	/* label of each class (label[k]) */
//    int nSV[MAX_NUM_OF_NODES_IN_RECORD];		/* number of SVs for each class (nSV[k]) */
//                                                        /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
//    /* XXX */
//    int free_sv;		/* 1 if svm_model is created by svm_load_model*/
//                                /* 0 if svm_model is created by svm_train */
//}svm_model;

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

double my_dot(__global const svm_node *px, __global const svm_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }			
    }
    return sum;
}

double kernel_linear(int i, int j, __global const svm_node* x, const int xW){
    return my_dot(&x[i * xW], &x[j * xW]);
}

double kernel_poly( int i, int j, __global const svm_node* x, const int xW, 
                    double gamma, double coef0, int degree){    
    return pow(gamma * my_dot(&x[i * xW], &x[j * xW]) + coef0, degree);
}

double kernel_rbf(  int i, int j, __global const svm_node* x, const int xW,
                    double gamma, __global double* x_square){   
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * my_dot(&x[i * xW], &x[j * xW])));
}

double kernel_sigmoid(  int i, int j, __global const svm_node* x, 
                        const int xW, double gamma, double coef0){
    return tanh(gamma * my_dot(&x[i * xW], &x[j * xW]) + coef0);
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
        data[realIndex] = y[i] * y[realIndex];
        switch (abs(kernel_type)){
        case LINEAR:
            data[realIndex] *= (float)kernel_linear(i, realIndex, x, xW);
            break;
        case POLY:
            data[realIndex] *= (float)kernel_poly(i, realIndex, x, xW, gamma, coef0, degree);
            break;
        case RBF:
            data[realIndex] *= (float)kernel_rbf(i, realIndex, x, xW, gamma, x_square);
            break;
        case SIGMOID:
            data[realIndex] *= (float)kernel_sigmoid(i, realIndex, x, xW, gamma, coef0);
            break;
        case PRECOMPUTED:
            data[realIndex] *= (float)kernel_precomputed(i, realIndex, x, xW);
            break;
        }
    }
}
//__kernel void SVR_Q_get_Q(  __global float* data, const int start, const int len, const int i, 
//                            const int kernel_type, __global svm_node* x, const int xW){
__kernel void svrQgetQ ( __global float* data, const int dataLen, const int start, const int len, const int i, 
                                const int kernel_type, __global svm_node* x, const int xW,
                                double gamma, double coef0, int degree, __global double* x_square){
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

//double k_function(const svm_node *x, const svm_node *y, const svm_parameter& param)
//{
//    switch(param.kernel_type)
//    {
//        case LINEAR:
//            return dot(x,y);
//        case POLY:
//            return powi(param.gamma * dot(x,y) + param.coef0, param.degree);
//        case RBF:
//        {
//            double sum = 0;
//            while(x->index != -1 && y->index !=-1)
//            {
//                if(x->index == y->index)
//                {
//                    double d = x->value - y->value;
//                    sum += d*d;
//                    ++x;
//                    ++y;
//                }
//                else
//                {
//                    if(x->index > y->index)
//                    {	
//                        sum += y->value * y->value;
//                        ++y;
//                    }
//                    else
//                    {
//                        sum += x->value * x->value;
//                        ++x;
//                    }
//                }
//            }
//
//            while(x->index != -1)
//            {
//                sum += x->value * x->value;
//                ++x;
//            }
//
//            while(y->index != -1)
//            {
//                sum += y->value * y->value;
//                ++y;
//            }
//
//            return exp(-param.gamma * sum);
//        }
//        case SIGMOID:
//            return tanh(param.gamma * dot(x,y) + param.coef0);
//        case PRECOMPUTED:  //x: test (validation), y: SV
//            return x[(int)(y->value)].value;
//        default:
//            return 0;  // Unreachable 
//    }
//}
//
//double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
//{
//    int i;
//    if(model->param.svm_type == ONE_CLASS ||
//       model->param.svm_type == EPSILON_SVR ||
//       model->param.svm_type == NU_SVR)
//    {
//        double *sv_coef = model->sv_coef[0];
//        double sum = 0;
//        for(i =0 ; i < model->l; i++)
//            sum += sv_coef[i] * k_function(x,model->SV[i],model->param);
//        sum -= model->rho[0];
//        *dec_values = sum;
//
//        if(model->param.svm_type == ONE_CLASS)
//            return (sum > 0) ? 1 : -1;
//        else
//            return sum;
//    }
//    else
//    {
//        int nr_class = model->nr_class;
//        int l = model->l;
//
//        double kvalue[MAX_NUM_OF_NODE_RECORDS]; //= Malloc(double,l);
//        for(i = 0; i < l; i++)
//            kvalue[i] = k_function(x, model->SV[i], model->param);
//
//        int start[MAX_NUM_OF_NODES_IN_RECORD] //= Malloc(int,nr_class);
//        start[0] = 0;
//        for(i = 1; i < nr_class; i++)
//            start[i] = start[i - 1] + model->nSV[i - 1];
//
//        int vote[MAX_NUM_OF_NODES_IN_RECORD];// = Malloc(int,nr_class);
//        for(i = 0; i < nr_class; i++)
//            vote[i] = 0;
//
//        int p=0;
//        for(i = 0; i < nr_class; i++)
//            for(int j = i + 1; j < nr_class; j++)
//            {
//                double sum = 0;
//                int si = start[i];
//                int sj = start[j];
//                int ci = model->nSV[i];
//                int cj = model->nSV[j];
//
//                int k;
//                double *coef1 = model->sv_coef[j-1];
//                double *coef2 = model->sv_coef[i];
//                for(k = 0; k < ci; k++)
//                        sum += coef1[si + k] * kvalue[si + k];
//                for(k = 0; k < cj; k++)
//                        sum += coef2[sj + k] * kvalue[sj + k];
//                sum -= model->rho[p];
//                dec_values[p] = sum;
//
//                if(dec_values[p] > 0)
//                        ++vote[i];
//                else
//                        ++vote[j];
//                p++;
//            }
//
//        int vote_max_idx = 0;
//        for(i = 1; i < nr_class; i++)
//            if(vote[i] > vote[vote_max_idx])
//                vote_max_idx = i;
//
//        //free(kvalue);
//        //free(start);
//        //free(vote);
//        return model->label[vote_max_idx];
//    }
//}
//
//double svm_predict(const svm_model *model, const svm_node *x)
//{
//    int nr_class = model->nr_class;
//    double dec_values[MAX_NUM_COEFS];
//    /**code below is commented due to avoid malloc and free
//    *we take more me but fuck, later will think something better
//    */
//    //if(model->param.svm_type == ONE_CLASS ||
//    //   model->param.svm_type == EPSILON_SVR ||
//    //   model->param.svm_type == NU_SVR)
//    //	dec_values = Malloc(double, 1);
//    //else 
//    //	dec_values = Malloc(double, nr_class * (nr_class - 1) / 2);
//    double pred_result = svm_predict_values(model, x, dec_values);
//    //free(dec_values);
//    return pred_result;
//}