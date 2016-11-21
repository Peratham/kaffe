#ifndef _KAFFE_VOLUME_H_
#define _KAFFE_VOLUME_H_

#include <algorithm>
#include <string>
#include <vector>

#include "proto/caffe.pb.h"

namespace kaffe {

template <typename Dtype>
class Volume {
public:
    explicit Volume(unsigned int number, unsigned int channel, unsigned int height, unsigned int width) {
        shape_.resize(4);
        shape_[0] = width;
        shape_[1] = height;
        shape_[2] = channel;
        shape_[3] = number;
        size_ = updateSize_();

        device_ = -1;
        data_ = new Dtype[size_];
    }
    explicit Volume(const std::vector<unsigned int> shape) {
        shape_ = shape;
        size_ = updateSize_();

        device_ = -1;
        data_ = new Dtype[size_];
    }

    ~Volume() {
        if (device_ == -1) {        // -1 : CPU
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
    Dtype* getData() {
        return data_;
    }

    bool cpu();
    bool gpu(unsigned int dev = 0);

private:
    size_t updateSize_() const {
        size_t totalSize = 0;
        for(size_t i = 0; i < shape_.size(); i++ ) {
            totalSize = totalSize * shape_[i];
        }
        return totalSize;
    }

protected:
    int device_;
    Dtype* data_;

    std::vector<unsigned int> shape_;
    size_t size_;
};  // class Volume

}  // namespace kaffe

#endif  // KAFFE_VOLUME_H_
