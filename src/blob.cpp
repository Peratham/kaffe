#include <assert.h>
#include <iostream>

#include <kaffe/blob.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
bool Blob<Dtype>::cpu() {
#ifdef USE_GPU
    if ( device_ == -1) {
        return true;
    }

    return false;
#else
    assert( device_ == -1);
    return true;
#endif
}

template <typename Dtype>
bool Blob<Dtype>::gpu(unsigned int dev) {
#ifndef USE_GPU
    assert( device_ == -1);
    return false;
#else

    return false;
#endif
}
  
template <typename Dtype>
bool Blob<Dtype>::copyFromProto(const caffe::BlobProto& proto) {
  // re-allocate memory as same as proto
  reset();
  
  shape_.resize(proto.shape().dim_size(), 0);
  for (int i = 0; i < proto.shape().dim_size(); ++i) {
    shape_[i] = (unsigned int)proto.shape().dim(i);
  }
  size_ = getSize();
  data_ = new Dtype[size_];
  
  // copy from proto (double or float)
  if (proto.double_data_size() > 0) {
    for (int i = 0; i < proto.double_data_size(); ++i) {
      data_[i] = proto.double_data(i);
    }
  } else {
    for (int i = 0; i < proto.data_size(); ++i) {
      data_[i] = proto.data(i);
    }
  }
  
  return true;
}

INSTANTIATE_CLASS(Blob);

} // namespace kaffe
