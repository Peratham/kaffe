#ifndef _KAFFE_NET_H_
#define _KAFFE_NET_H_

#include <algorithm>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include <kaffe/blob.h>
#include <kaffe/layer.h>
#include <kaffe/engine.h>
#include <kaffe/proto/caffe.pb.h>

//
// DAG (directed acyclic graph) implement
//

namespace kaffe {

template <typename Dtype>
class Net {
public:
    Net(const std::string& param_file, const std::string& weight_file, const KAFFE_ENGINE engine) {
      loadParamFile(param_file);
      loadWeightFile(weight_file);
      engine_ = Engine<Dtype>::createEngine( engine );
      assert(engine_ != NULL);
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

    Dtype Forward();

    Blob<Dtype>* getBlob(const std::string& name) {
      if ( blobsMap_.find(name) == blobsMap_.end() ) {
        return NULL;
      }

      return blobs_[  blobsMap_[name] ];
    }

private:
    void loadParamFile(const std::string& param_file);
    void loadWeightFile(const std::string& weight_file);
    Layer<Dtype>* createLayer(const caffe::LayerParameter& param);

protected:
    std::vector<Layer<Dtype>*> layers_;
    std::vector<Blob<Dtype>*> blobs_;       // blobs_[0] means input

    std::vector< std::vector<Blob<Dtype>*> > bottoms_;
    std::vector< std::vector<Blob<Dtype>*> > tops_;

    //fast access
    std::map<std::string, size_t> layersMap_;
    std::map<std::string, size_t> blobsMap_;

    Engine<Dtype>* engine_;
};

} // namespace kaffe
#endif
