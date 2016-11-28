#include <assert.h>

#include <kaffe/blob.h>
#include <kaffe/layer.h>
#include <kaffe/engine.h>
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
Dtype InnerProductLayer<Dtype>::Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                                       const std::vector<Blob<Dtype>*>& top) {
  assert(bottom.size() == 1);
  assert(top.size() == 1);
  
  // check input
  std::vector<unsigned int> inputShape = bottom[0]->shape();
  std::vector<unsigned int> newInputShape(2);
  if ( inputShape.size() == 2) {                                          // 2D  batch x size
    newInputShape = inputShape;
  } else if ( inputShape.size() == 4) {
    newInputShape[0] = inputShape[0];
    newInputShape[1] = inputShape[1] * inputShape[2] * inputShape[3];     // 4D  batch x size ( C x H x W)
  } else {
    assert(false);
  }
  bottom[0]->reset(newInputShape,true);
 
  // check filter
  assert( this->blobs_[0]->shape()[1] == newInputShape[1] );
  
  // check output
  std::vector<unsigned int> outputShape(2);
  outputShape[0] = newInputShape[0];                                      // output number
  outputShape[1] = this->blobs_[0]->shape()[0];                           // output vector size
  if ( outputShape[0] * outputShape[1] != top[0]->size() ) {
    top[0]->reset(outputShape);
  }
  
  // do product
  eng->product(*bottom[0], *(this->blobs_[0]),  *(top[0]));
  
  // restore input shape
  bottom[0]->reset(newInputShape,true);
  return 0.0;
}

INSTANTIATE_CLASS(InnerProductLayer);

}
