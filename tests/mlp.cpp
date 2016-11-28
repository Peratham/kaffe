#include <string>
#include <iostream>

#include <kaffe/net.h>
#include <kaffe/blob.h>

typedef float Dtype;
using namespace kaffe;

int main(int argc, const char* argv[]) {
  
  if ( argc < 3) {
    std::cout << "Input network and weight." << std::endl;
    return -1;
  }
  
  const std::string network = argv[1];
  const std::string weight = argv[2];
  
  std::vector<unsigned int> shape;
  shape.resize(2);
  shape[0] = 1;
  shape[1] = 2;
  
  Net<Dtype>* net = new Net<Dtype>(network, weight, KE_EIGEN_CPU);
  Blob<Dtype>* data = net->getBlob("data");
  data->reset(shape);
  data->data()[0] = 0.5;
  data->data()[1] = 0.5;
  
  net->Forward();
  
  data = net->getBlob("prob");
  std::cout << "##########:" << data->size() << std::endl;
  
  return 0;
}

