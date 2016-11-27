#ifndef _KAFFE_IM2COL_H_
#define _KAFFE_IM2COL_H_

#include <kaffe/blob.h>
#include <Eigen/Dense>

namespace kaffe {

template <typename Dtype>
class im2col {
public:
static Eigen::Matrix<Dtype, -1, -1> im2col_cpu(Blob<Dtype>& src, const int numIndex,
                                               const int kernel_w, const int kernel_h,
                                               const int stride_w, const int stride_h,
                                               const int pad_w, const int pad_h);

static Eigen::Matrix<Dtype, -1, -1> filter2col_cpu(Blob<Dtype>& filter, const int number,
                                                   const int targetHeight, const int targetWidth);

};

}
#endif
