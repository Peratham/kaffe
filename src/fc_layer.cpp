#include <assert.h>

#include <kaffe/blob.h>
#include <kaffe/layer.h>
#include <kaffe/fc_layer.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
InnerProductLayer<Dtype>::InnerProductLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) {
  std::vector<unsigned int> shape;
  
  assert( param.has_inner_product_param());
  fcParam = param.inner_product_param();
  
  this->blobs_.resize(2);
  this->blobs_[0] = new Blob<Dtype>();    // for weights
  this->blobs_[1] = new Blob<Dtype>();    // for bias
}

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Forward(const std::vector<Blob<Dtype>*>& bottom,
                                       const std::vector<Blob<Dtype>*>& top) {
  return 0.0;
}

INSTANTIATE_CLASS(InnerProductLayer);

}
