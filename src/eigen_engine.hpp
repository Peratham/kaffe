#ifndef _KAFFE_EIGEN_ENGINE_H_
#define _KAFFE_EIGEN_ENGINE_H_

#include <Eigen/Dense>
#include <kaffe/engine.h>

namespace kaffe {

#define MATRIX Eigen::Matrix<Dtype, -1, -1>

template <typename Dtype>
class EigenEngine : public Engine<Dtype> {
public:
    explicit EigenEngine(const int device) : device_(device) {

    }
    virtual ~EigenEngine(){}
  
    virtual void product(Blob<Dtype>& a, Blob<Dtype>& b, Blob<Dtype>& c);
    virtual void conv(Blob<Dtype>& input, Blob<Dtype>& filter, Blob<Dtype>& output,
                       const int kernel_h, const int kernel_w,
                       const int stride_h, const int stride_w,
                       const int pad_h, const int pad_w);
private:
    Eigen::Matrix<Dtype, -1, -1> im2col_cpu(Blob<Dtype>& src, const int numIndex,
                                            const int kernel_w, const int kernel_h,
                                            const int stride_w, const int stride_h,
                                            const int pad_w, const int pad_h);

protected:
    const int device_;        //-1: CPU
};

}
#endif
