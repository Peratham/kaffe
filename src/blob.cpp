#include <assert.h>

#include <kaffe/blob.h>

#include "common.hpp"

namespace kaffe {

template <typename Dtype>
bool Blob<Dtype>::cpu() {
#ifdef USE_GPU
    if ( device_ == -1) {
        return true;
    }

    return false;
#else
    assert( device_ == -1);
    return true;
#endif
}

template <typename Dtype>
bool Blob<Dtype>::gpu(unsigned int dev) {
#ifndef USE_GPU
    assert( device_ == -1);
    return false;
#else

    return false;
#endif
}

INSTANTIATE_CLASS(Blob);

} // namespace kaffe
