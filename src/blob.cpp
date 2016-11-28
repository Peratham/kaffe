#include <assert.h>
#include <iostream>

#include <kaffe/blob.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
bool Blob<Dtype>::copyFromProto(const caffe::BlobProto& proto) {
  // re-allocate memory as same as proto
  std::vector<unsigned int> newShape(proto.shape().dim_size());
  for (int i = 0; i < proto.shape().dim_size(); ++i) {
    newShape[i] = (unsigned int)proto.shape().dim(i);
  }
  reset(newShape);
  
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
