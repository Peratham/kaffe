#ifndef _KAFFE_LAYER_H_
#define _KAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include <kaffe/blob.h>
#include <kaffe/engine.h>
#include <kaffe/proto/caffe.pb.h>

namespace kaffe {

template <typename Dtype>
class Layer {
public:
    explicit Layer(const caffe::LayerParameter& param) : layer_param_(param) {
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

    const caffe::LayerParameter& param() {
      return layer_param_;
    }

    virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
            const std::vector<Blob<Dtype>*>& top) = 0;

protected:
    caffe::LayerParameter layer_param_;
    std::vector<Blob<Dtype>* > blobs_;       // weight and bias
};

template <typename Dtype>
class InputLayer : public Layer<Dtype> {
public:
  explicit InputLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) {
    assert( param.has_input_param() );
    inputParam = param.input_param();
    
  }
  
  virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                    const std::vector<Blob<Dtype>*>& top) {
    assert( inputParam.shape_size() == top.size());
    for(size_t i = 0; i < top.size(); i++) {
      assert( top[i]->size() > 0);
    }
    return 0.0;
  }
  
private:
  caffe::InputParameter inputParam;
};

template <typename Dtype>
class ReLULayer : public Layer<Dtype> {
public:
explicit ReLULayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                        const std::vector<Blob<Dtype>*>& top) {
    // Do nothing
    return 0.0;
}
};

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
public:
explicit PoolingLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                        const std::vector<Blob<Dtype>*>& top) {
    // Do nothing
    return 0.0;
}
};

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
explicit SoftmaxLayer(const caffe::LayerParameter& param) : Layer<Dtype>(param) { }
virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
                        const std::vector<Blob<Dtype>*>& top) {
    // Do nothing
    return 0.0;
}
};

} // namespace kaffe
#endif
