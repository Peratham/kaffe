#include <iostream>

#include "common.hpp"
#include "eigen_engine.hpp"

namespace kaffe {

template <typename Dtype>
void EigenEngine<Dtype>::conv(Blob<Dtype>& input, Blob<Dtype>& filter, Blob<Dtype>& output,
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w) {

}

template <typename Dtype>
void EigenEngine<Dtype>::product(Blob<Dtype>& a, Blob<Dtype>& b, Blob<Dtype>& c) {
  const int aRow = a.shape()[0];
  const int aCol = a.shape()[1];
  const int bRow = b.shape()[0];
  const int bCol = b.shape()[1];
  const int cRow = c.shape()[0];
  const int cCol = c.shape()[1];

  assert(aCol == bCol);
  assert(cRow == aRow);
  assert(cCol == bRow);
  
  Eigen::Map<Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor> >  A(a.data(), aRow, aCol);
  Eigen::Map<Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor> >  B(b.data(), bRow, bCol);
  Eigen::Map<Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor> >  C(c.data(), cRow, cCol);

  C = A * B.transpose();
}
  
INSTANTIATE_CLASS(EigenEngine);

}
