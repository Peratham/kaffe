#ifndef _KAFFE_BLOB_H_
#define _KAFFE_BLOB_H_

#include <algorithm>
#include <string>
#include <vector>

#include <kaffe/proto/caffe.pb.h>

namespace kaffe {

template <typename Dtype>
class Blob {
public:
    explicit Blob(const std::vector<unsigned int>& shape) {
      shape_ = shape;
      size_ = getSize();

      data_ = new Dtype[size_];
    }
    explicit Blob(unsigned int number, unsigned int channel, unsigned int height, unsigned int width) {
        shape_.resize(4);
        shape_[0] = number;
        shape_[1] = channel;
        shape_[2] = height;
        shape_[3] = width;
        size_ = getSize();

        data_ = new Dtype[size_];
    }

    explicit Blob(unsigned int number, unsigned int length) {
      shape_.resize(2);
      shape_[0] = number;
      shape_[1] = length;
      
      size_ = getSize();

      data_ = new Dtype[size_];
    }

    explicit Blob() {
        shape_.clear();
        size_ = 0;
        data_ = NULL;
    }

    ~Blob() {
        if (data_ != NULL) {
            delete data_;
            return;
        }
    }
    size_t size() const {
        return size_;
    }
    const std::vector<unsigned int>& shape() const {
        return shape_;
    }
    Dtype* data() {
        return data_;
    }
  
    void reshape(const std::vector<unsigned int>& shape) {
      int newSize = 1;
      for(size_t i = 0; i < shape.size(); i++) {
        newSize = newSize * shape[i];
      }
      assert(newSize == size_);
      shape_ = shape;
    }

    void reset(const std::vector<unsigned int>& shape, const bool reuse = false) {
      if ( reuse ) {
        int newSize = 1;
        for(size_t i = 0; i < shape.size(); i++) {
          newSize = newSize * shape[i];
        }
        assert(newSize == size_);
        shape_ = shape;
        return;
      }
      
      if ( data_ != NULL) {
        delete data_;
      }
      shape_ = shape;
      size_ = getSize();
      data_ = new Dtype[size_];
    }

    bool copyFromProto(const caffe::BlobProto& blobProto);

private:
    size_t getSize() const {
        size_t totalSize = 1;
        for(size_t i = 0; i < shape_.size(); i++ ) {
            totalSize = totalSize * shape_[i];
        }
        return totalSize;
    }

protected:
    Dtype* data_;
  
    std::vector<unsigned int> shape_;
    size_t size_;
};  // class Blob

}  // namespace kaffe

#endif  // KAFFE_VOLUME_H_
