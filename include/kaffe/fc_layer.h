#ifndef _KAFFE_FC_LAYER_H_
#define _KAFFE_FC_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include <kaffe/blob.h>
#include <kaffe/layer.h>

#include "proto/caffe.pb.h"

namespace kaffe {

template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
public:
  explicit InnerProductLayer(const caffe::LayerParameter& param);
  virtual Dtype Forward(Engine<Dtype>* eng, const std::vector<Blob<Dtype>*>& bottom,
            const std::vector<Blob<Dtype>*>& top);

private:
  caffe::InnerProductParameter fcParam;
};

}
#endif
