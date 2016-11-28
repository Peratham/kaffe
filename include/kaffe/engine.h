#ifndef _KAFFE_ENGINE_H_
#define _KAFFE_ENGINE_H_

#include <kaffe/blob.h>

namespace kaffe {

enum KAFFE_ENGINE {
    KE_EIGEN_CPU = 0,
#ifdef USE_GPU
    KE_EIGEN_GPU = 1,
#endif
};

template <typename Dtype>
class Engine {
public:
    explicit Engine(){}
    virtual ~Engine(){}

    virtual void product(Blob<Dtype>& a, Blob<Dtype>& b, Blob<Dtype>& c) = 0;
    virtual void conv(Blob<Dtype>& input, Blob<Dtype>& filter, Blob<Dtype>& output,
                       const int kernel_h, const int kernel_w,
                       const int stride_h, const int stride_w,
                       const int pad_h, const int pad_w) = 0;

public:
    static Engine* createEngine(const KAFFE_ENGINE engine);
};
}
#endif

