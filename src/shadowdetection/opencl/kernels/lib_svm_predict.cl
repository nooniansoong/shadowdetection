#pragma OPENCL EXTENSION cl_khr_fp64: enable

enum { C_SVC = 0, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR = 0, POLY, RBF, SIGMOID, PRECOMPUTED };           /* kernel_type */

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
}svm_parameter;

typedef struct _svm_model
{
    svm_parameter* param;	/* parameter */
    int nr_class;		/* number of classes, = 2 in regression/one class svm */
    int svsLength;		/* total #SV */
    int svsWidth;
    //switch to one dimension
    __global const svm_node* SV;
    //switch to one dimension    
    //dimensions are l: nr_class - 1, w: svsLength
    __global const double* sv_coef;
    //dimension is: nr_class * (nr_class - 1) / 2;
    __global const double* rho;
    //dimension is: nr_class * (nr_class - 1) / 2;
    double* probA;		/* pariwise probability information */
    //dimension is: nr_class * (nr_class - 1) / 2;
    double* probB;
    //passing null
    int* sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

    /* for classification only */
    //dimension is: nr_class
    __global const int* label;	/* label of each class (label[k]) */
    //dimension is: nr_class
    __global const int* nSV;    /* number of SVs for each class (nSV[k]) */
                                /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    /* XXX */
    int free_sv;		/* 1 if svm_model is created by svm_load_model*/
                                /* 0 if svm_model is created by svm_train */
}svm_model;

double my_dot(const svm_node *px, __global const svm_node *py)
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

double kfunction_rbf(const svm_node* x, __global const svm_node* y, double gamma){
    double sum = 0;        
    while (x->index != -1 && y->index != -1) {
        if (x->index == y->index) {
            double d = x->value - y->value;
            sum += d * d;
            ++x;
            ++y;
        } else {
            if (x->index > y->index) {
                sum += y->value * y->value;
                ++y;
            } else {
                sum += x->value * x->value;
                ++x;
            }
        }
    }

    while (x->index != -1) {
        sum += x->value * x->value;
        ++x;
    }

    while (y->index != -1) {
        sum += y->value * y->value;
        ++y;
    }

    return exp(gamma * sum);
}

double k_function(const svm_node *x, __global const svm_node *y, const svm_parameter* param) {
    switch (param->kernel_type) {
        case LINEAR:
            return my_dot(x, y);
        case POLY:
            return pow(param->gamma * my_dot(x, y) + param->coef0, param->degree);
        case RBF:
        {
            return kfunction_rbf(x, y, param->gamma);
        }
        case SIGMOID:
            return tanh(param->gamma * my_dot(x, y) + param->coef0);
        case PRECOMPUTED: //x: test (validation), y: SV
            return x[(int)(y->value)].value;
        default:
            return 0; // Unreachable 
    }
}

double svm_predict_values(  const svm_model *model, const svm_node *x, double* dec_values,
                            __local double* kvalue, __local int* start, __local int* vote)
{
    int i;
    if(model->param->svm_type == ONE_CLASS ||
       model->param->svm_type == EPSILON_SVR ||
       model->param->svm_type == NU_SVR)
    {
        //sv_coef[0] this is the same
        __global const double *sv_coef = model->sv_coef; 
        double sum = 0;
        for(i =0 ; i < model->svsLength; i++)
            sum += sv_coef[i] * k_function(x, &model->SV[i * model->svsWidth], model->param);
        sum -= model->rho[0];
        *dec_values = sum;

        if(model->param->svm_type == ONE_CLASS)
            return (sum > 0) ? 1 : -1;
        else
            return sum;
    }
    else
    {
        int nr_class = model->nr_class;
        int l = model->svsLength;
        
        for(i = 0; i < l; i++)
            kvalue[i] = k_function(x, &model->SV[i * model->svsWidth], model->param);
        
        start[0] = 0;
        for(i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + model->nSV[i - 1];
        
        for(i = 0; i < nr_class; i++)
            vote[i] = 0;

        int p = 0;
        for(i = 0; i < nr_class; i++)
            for(int j = i + 1; j < nr_class; j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];

                int k;
                __global const double *coef1 = &(model->sv_coef[(j - 1) * model->svsLength]);
                __global const double *coef2 = &(model->sv_coef[i * model->svsLength]);
                for(k = 0; k < ci; k++)
                    sum += coef1[si + k] * kvalue[si + k];
                for(k = 0; k < cj; k++)
                    sum += coef2[sj + k] * kvalue[sj + k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        int vote_max_idx = 0;
        for(i = 1; i < nr_class; i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;
        
        return model->label[vote_max_idx];
    }
}

//dec_values size = 1 or nr_class * (nr_class - 1) / 2, needs to be just allocated
//kvalue size = svsLength, needs to be just allocated
//start size = nr_class, needs to be just allocated
//vote size = nr_class, needs to be just allocated
double svm_predict( const svm_model *model, const svm_node *x, double* dec_values,
                    __local double* kvalue, __local int* start, __local int* vote)
{        
    double pred_result = svm_predict_values(model, x, dec_values, 
                                            kvalue, start, vote);    
    return pred_result;
}

__kernel void predict(  __global const svm_node *x, const uint xLen, const int nr_class,
                        const int svsLength, const int svsWidth, __global const svm_node* SV,
                        __global const double* sv_coef, __global const double* rho,
                        __global const int* label, __global const int* nSV, const int free_sv,
                        //parameter args
                        const int svm_type, const int kernel_type, const int degree,
                        const double gamma, const double coef0, __global const int* results){
    svm_model model;
    model.nr_class = nr_class;
    model.svsLength = svsLength;
    model.svsWidth = svsWidth;
    model.SV = SV;
    model.sv_coef = sv_coef;
    model.rho = rho;
    model.probA = 0;
    model.probB = 0;
    model.sv_indices = 0;
    model.label = label;
    model.nSV = nSV;
    model.free_sv = free_sv;
                                    
    svm_parameter parameter;
    parameter.svm_type = svm_type;
    parameter.kernel_type = kernel_type;
    parameter.degree = degree;
    parameter.gamma = gamma;
    parameter.coef0 = coef0;    
    
    model.param = &parameter;
}