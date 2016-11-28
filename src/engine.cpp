#include <assert.h>
#include <iostream>

#include <kaffe/engine.h>

#include "common.hpp"
#include "eigen_engine.hpp"

namespace kaffe {

// top factory to create engine
template <typename Dtype>
Engine<Dtype>* Engine<Dtype>::createEngine(const KAFFE_ENGINE engine) {
    Engine<Dtype>* ret = NULL;
    if ( engine == KE_EIGEN_CPU) {
        ret = new EigenEngine<Dtype>(-1);
    }
    return ret;
}


INSTANTIATE_CLASS(Engine);

}

