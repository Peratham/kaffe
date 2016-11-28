#include <iostream>
#include <assert.h>

#include <kaffe/blob.h>
#include <kaffe/layer.h>
#include <kaffe/conv_layer.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
ConvolutionLayer<Dtype>::ConvolutionLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) {
  std::vector<unsigned int> shape;

  assert( param.has_convolution_param() );
  convParam = param.convolution_param();

  // un-support features, so far
  assert(convParam.has_axis() == 0);
  assert(convParam.has_group() == 0);
  assert(convParam.dilation_size() == 0);

  this->blobs_.resize(2);
  this->blobs_[0] = new Blob<Dtype>();    // for weights
  this->blobs_[1] = new Blob<Dtype>();    // for bias
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                                       const std::vector<Blob<Dtype>*>& top) {
  return 0.0;
}

INSTANTIATE_CLASS(ConvolutionLayer);

}
