#include <assert.h>
#include <iostream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <kaffe/net.h>
#include <kaffe/layer.h>
#include <kaffe/conv_layer.h>
#include <kaffe/fc_layer.h>

#include "common.hpp"
#include "upgrade_proto.hpp"

namespace kaffe {

template <typename Dtype>
Layer<Dtype>* Net<Dtype>::createLayer(const caffe::LayerParameter& param) {
  Layer<Dtype>* layer = NULL;

  if ( param.type() == "Convolution" ) {
    layer = new ConvolutionLayer<Dtype>(param);
  } else if ( param.type() == "InnerProduct" ) {
    layer = new InnerProductLayer<Dtype>(param);
  } else if ( param.type() == "Input" ) {
    layer = new InputLayer<Dtype>(param);
  } else if ( param.type() == "Pooling" ) {
    layer = new PoolingLayer<Dtype>(param);
  } else if ( param.type() == "ReLU" ) {
    layer = new ReLULayer<Dtype>(param);
  } else if ( param.type() == "Softmax" ) {
    layer = new SoftmaxLayer<Dtype>(param);
  } else {
    assert(false);
  }

  return layer;
}

template <typename Dtype>
void Net<Dtype>::loadParamFile(const std::string& param_file) {
    caffe::NetParameter param;
    caffe::ReadNetParamsFromTextFileOrDie(param_file, &param);

    // building and check DAG
    for(int i = 0; i < param.layer_size(); i++) {
      const caffe::LayerParameter layer_param = param.layer(i);

      assert( layersMap_.find( layer_param.name() ) == layersMap_.end() );
      Layer<Dtype>* layer = createLayer(layer_param);
      layers_.push_back(layer);
      layersMap_[ layer_param.name() ] = layers_.size() - 1;

      std::vector<Blob<Dtype>*> newBottom;
      for(int i = 0; i < layer_param.bottom_size(); i++) {
        std::string blobName = layer_param.bottom(i);
        assert( blobsMap_.find(blobName) != blobsMap_.end());
        newBottom.push_back( blobs_[blobsMap_[blobName]] );
      }
      bottoms_.push_back(newBottom);

      std::vector<Blob<Dtype>*> newTop;
      for(int i = 0; i < layer_param.top_size(); i++) {
        std::string blobName = layer_param.top(i);
        if ( blobsMap_.find(blobName) == blobsMap_.end()) {
          Blob<Dtype>* blob = new Blob<Dtype>();
          blobs_.push_back(blob);
          blobsMap_[blobName] = blobs_.size() - 1;
        } else {
          // TODO checking in-place computing
        }
        newTop.push_back( blobs_[blobsMap_[blobName]] );
      }
      tops_.push_back( newTop);
    }
}

template <typename Dtype>
void Net<Dtype>::loadWeightFile(const std::string& weight_file) {
  caffe::NetParameter param;
  caffe::ReadNetParamsFromBinaryFileOrDie(weight_file, &param);

  for(int i = 0; i < param.layer_size(); i++) {

    const caffe::LayerParameter layer_param = param.layer(i);
    if ( layer_param.blobs_size() == 0 ) {
      continue;
    }
    if ( layersMap_.find(layer_param.name()) == layersMap_.end() ) {
      continue;
    }
    const size_t layer_index = layersMap_[ layer_param.name() ];

    std::vector<Blob<Dtype>* > layerBlobs = layers_[layer_index]->blobs();
    std::cout << layer_param.name() << std::endl;
    assert( layerBlobs.size() == layer_param.blobs_size() );


    for(int j = 0; j < layer_param.blobs_size(); j++) {
      bool success = layerBlobs[j]->copyFromProto( layer_param.blobs(j));
      assert(success == true);
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::Forward() {
  Dtype loss = 0.0;
  for(size_t i = 0; i < layers_.size(); i++) {
    loss = loss + layers_[i]->Forward(engine_, bottoms_[i], tops_[i]);
  }
  return loss;
}

INSTANTIATE_CLASS(Net);
} // namespace kaffe
