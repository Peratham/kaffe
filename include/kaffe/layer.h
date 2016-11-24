#ifndef _KAFFE_LAYER_H_
#define _KAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include <kaffe/blob.h>
#include "proto/caffe.pb.h"

namespace kaffe {

template <typename Dtype>
class Layer {
public:
    explicit Layer(const caffe::LayerParameter& param) : layer_param_(param) {
        device_ = -1;
        blobs_.clear();
    }

    virtual ~Layer() {
        for (size_t i = 0; i < blobs_.size(); ++i) {
            delete blobs_[i];
            blobs_[i] = NULL;
        }
    };
  
    std::vector<Blob<Dtype>* > blobs() {
      return blobs_;
    }

    bool cpu();
    bool gpu(unsigned int dev = 0);
    virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
            const std::vector<Blob<Dtype>*>& top) = 0;

protected:
    int device_;

    caffe::LayerParameter layer_param_;
    std::vector<Blob<Dtype>* > blobs_;       // weight and bias
};

  template <typename Dtype>
  class InputLayer : public Layer<Dtype> {
  public:
    explicit InputLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
    virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
                        const std::vector<Blob<Dtype>*>& top) {
      // Do nothing
      return 0.0;
    }
  };

  template <typename Dtype>
  class ReLULayer : public Layer<Dtype> {
  public:
    explicit ReLULayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
    virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
                          const std::vector<Blob<Dtype>*>& top) {
      // Do nothing
      return 0.0;
    }
  };

  template <typename Dtype>
  class PoolingLayer : public Layer<Dtype> {
  public:
    explicit PoolingLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
    virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
                          const std::vector<Blob<Dtype>*>& top) {
      // Do nothing
      return 0.0;
    }
  };

  template <typename Dtype>
  class SoftmaxLayer : public Layer<Dtype> {
  public:
    explicit SoftmaxLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
    virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
                          const std::vector<Blob<Dtype>*>& top) {
      // Do nothing
      return 0.0;
    }
  };

} // namespace kaffe
#endif
