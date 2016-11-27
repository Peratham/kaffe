#ifndef _KAFFE_BLOB_H_
#define _KAFFE_BLOB_H_

#include <algorithm>
#include <string>
#include <vector>

#include "proto/caffe.pb.h"

namespace kaffe {

template <typename Dtype>
class Blob {
public:
    explicit Blob(const std::vector<unsigned int>& shape) {
      shape_ = shape;
      size_ = getSize();
    
      device_ = -1;
      data_ = new Dtype[size_];
    }
    explicit Blob(unsigned int number, unsigned int channel, unsigned int height, unsigned int width) {
        shape_.resize(4);
        shape_[0] = width;
        shape_[1] = height;
        shape_[2] = channel;
        shape_[3] = number;
        size_ = getSize();

        device_ = -1;
        data_ = new Dtype[size_];
    }
  
    explicit Blob(unsigned int channel, unsigned int height, unsigned int width) {
      shape_.resize(3);
      shape_[0] = width;
      shape_[1] = height;
      shape_[2] = channel;
      size_ = getSize();
    
      device_ = -1;
      data_ = new Dtype[size_];
    }
  
    explicit Blob() {
        shape_.clear();
        size_ = 0;
        device_ = -1;
        data_ = NULL;
    }

    ~Blob() {
        if (device_ == -1 && data_ != NULL) {        // -1 : CPU
            delete data_;
            return;
        }
#ifdef USE_GPU

#endif
    }
    size_t size() const {
        return size_;
    }
    std::vector<unsigned int> shape() const {
        return shape_;
    }
    int device() const {
        return device_;
    }
    Dtype* data() {
        return data_;
    }
  
    void reset(const std::vector<unsigned int>& shape) {
      reset();
      shape_ = shape;
      size_ = getSize();
      
      device_ = -1;
      data_ = new Dtype[size_];
    }
  
    bool copyFromProto(const caffe::BlobProto& blobProto);
    bool cpu();
    bool gpu(unsigned int dev = 0);

private:
    size_t getSize() const {
        size_t totalSize = 1;
        for(size_t i = 0; i < shape_.size(); i++ ) {
            totalSize = totalSize * shape_[i];
        }
        return totalSize;
    }
  
    void reset() {
      if ( data_ != NULL) {
        delete data_;
      }
      
      device_ = -1;
      shape_.clear();
      size_ = 0;
      data_ = NULL;
   }

protected:
    int device_;
    Dtype* data_;

    std::vector<unsigned int> shape_;
    size_t size_;
};  // class Blob

}  // namespace kaffe

#endif  // KAFFE_VOLUME_H_
