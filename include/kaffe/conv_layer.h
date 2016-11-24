#ifndef _KAFFE_CONV_LAYER_H_
#define _KAFFE_CONV_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include <kaffe/blob.h>
#include <kaffe/layer.h>

#include "proto/caffe.pb.h"

namespace kaffe {

template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
public:
  explicit ConvolutionLayer(const caffe::LayerParameter& param);
  virtual Dtype Forward(const std::vector<Blob<Dtype>*>& bottom,
            const std::vector<Blob<Dtype>*>& top);

private:
  caffe::ConvolutionParameter convParam;
  
};

}
#endif
