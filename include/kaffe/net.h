#ifndef _KAFFE_NET_H_
#define _KAFFE_NET_H_

#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include <kaffe/blob.h>
#include <kaffe/layer.h>

#include "proto/caffe.pb.h"

//
// DAG (directed acyclic graph) implement
//

namespace kaffe {

class DAGNode;

template <typename Dtype>
class Net {
public:
    Net(const std::string& param_file, const std::string& weight_file) {
      loadFromFiles(param_file, weight_file);
    }
  
    ~Net() {
      
      for(size_t i = 0; i < layers_.size(); i++) {
        delete layers_[i];
        layers_[i] = NULL;
      }
      
      for(size_t i = 0; i < blobs_.size(); i++) {
        delete blobs_[i];
        blobs_[i];
      }
      
    };
  
    Blob<Dtype>* getBlob(const std::string& name) {
      if ( blobsMap_.find(name) == blobsMap_.end() ) {
        return NULL;
      }
    
      return blobs_[  blobsMap_[name] ];
    }
  
    Dtype Forward();
    bool cpu();
    bool gpu(unsigned int dev = 0);

private:
    void loadFromFiles(const std::string& param_file, const std::string& weight_file);
    Layer<Dtype>* createLayer(const caffe::LayerParameter& param);

protected:
    int device_;
    std::vector<Layer<Dtype>*> layers_;
    std::vector<Blob<Dtype>*> blobs_;       // blobs_[0] means input
  
    std::vector< std::vector<Blob<Dtype>*> > bottoms_;
    std::vector< std::vector<Blob<Dtype>*> > tops_;
  
    //fast access
    std::map<std::string, size_t> layersMap_;
    std::map<std::string, size_t> blobsMap_;

};

} // namespace kaffe
#endif
