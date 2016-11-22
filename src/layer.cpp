#include <assert.h>

#include <kaffe/layer.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
bool Layer<Dtype>::cpu() {
    return true;
}

template <typename Dtype>
bool Layer<Dtype>::gpu(unsigned int dev) {
    return false;
}

INSTANTIATE_CLASS(Layer);

} // namespace kaffe
