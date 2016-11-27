#include <string>
#include <iostream>

#include <kaffe/net.h>
#include <kaffe/blob.h>
#include <im2col.hpp>

typedef float Dtype;
using namespace kaffe;

int main(int argc, const char* argv[]) {
  Blob<Dtype> blob(2, 4, 4);
  
  std::vector<unsigned int> shape = blob.shape();
  Dtype* d = blob.data();
  
  int n = 0;
  for(int c = 0; c < shape[2]; c++) {
    for(int row = 0; row < shape[1]; row ++) {
      for( int col = 0; col < shape[0]; col++) {
        d[ c * shape[1]*shape[0] + row * shape[0] + col] = n;
        n++;
      }
    }
  }
  
  Eigen::Matrix<Dtype, -1, -1> colImage = im2col<Dtype>::im2col_cpu(blob, 0, 2, 2, 1, 1, 0, 0);
  
  std::cout << colImage << std::endl;
  std::cout << "###############" << std::endl;
  
  
  Blob<Dtype> filter(2, 2, 2);
  for(int i = 0; i < filter.size(); i++) {
    filter.data()[i] = 0;
  }
  Eigen::Matrix<Dtype, -1, -1> colFilter = im2col<Dtype>::filter2col_cpu(filter, 2, 3, 3);
  
  std::cout << colFilter << std::endl;
  
}

