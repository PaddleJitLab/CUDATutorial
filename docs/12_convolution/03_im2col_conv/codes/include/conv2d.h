#ifndef __CONV2D_FWD_HEADER__
#define __CONV2D_FWD_HEADER__

#define __in__
#define __out__
#define __in_out__

typedef struct
{
    float *in;      // 输入数据地址
    float *weight;  // 权值数据地址
    float *out;     // 输出数据地址
    unsigned int n; // batch szie              default value 1
    unsigned int c; // channel number          default value 32
    unsigned int h; // 数据高                   default value 32
    unsigned int w; // 数据宽                   default value 32
    unsigned int k; // 卷积核数量                default value 32
    unsigned int r; // 卷积核高                  default value 1
    unsigned int s; // 卷积核宽                  default value 1
    unsigned int u; // 卷积在高方向上的步长        default value 1
    unsigned int v; // 卷积在宽方向上的步长        default value 1
    unsigned int p; // 卷积在高方向上的补边        default value 0
    unsigned int q; // 卷积在宽方向上的补边        default value 0
} problem_t;

typedef struct
{
    unsigned int blockx;        // blockx  number
    unsigned int blocky;        // blocky  number
    unsigned int blockz;        // blockz  number
    unsigned int threadx;       // threadx number per block
    unsigned int thready;       // thready number per block
    unsigned int threadz;       // threadz number per block
    unsigned int dynmicLdsSize; // 动态分配的lds大小，如果不使用动态分配的lds，则该值为0；
    void *kernelPtr;            // kernel ptr
} kernelInfo_t;

int getParamsize(__in__ problem_t *problem, __out__ int *paramSize);
int getkernelInfo(__in__ problem_t *problem, __out__ kernelInfo_t *kernelInfo, __in_out__ void *param);

#endif