#include <assert.h>

#include "common.hpp"
#include "im2col.hpp"

namespace kaffe {

// more detail https://www.zhihu.com/question/28385679
template <typename Dtype>
Eigen::Matrix<Dtype, -1, -1> im2col<Dtype>::im2col_cpu(Blob<Dtype>& src, const int numIndex,
                                                         const int kernel_w, const int kernel_h,
                                                         const int stride_w, const int stride_h,
                                                         const int pad_w, const int pad_h) {
  std::vector<unsigned int> shape = src.shape();
  
  // check input size
  assert(src.device() == -1);
  if ( shape.size() == 4 ) {
    assert( numIndex >= 0 && numIndex < shape[3]);
  }
  if ( shape.size() == 3) {
    assert(numIndex == 0);
  }
  
  unsigned int imgWidth = shape[0];
  unsigned int imgHeight = shape[1];
  unsigned int imgChannel = shape[2];
  
  const int targetWidth = (imgWidth + pad_w * 2 - kernel_w) / stride_w + 1;
  const int targetHeight = (imgHeight + pad_h * 2 - kernel_h) / stride_h + 1;
  
  Eigen::Matrix<Dtype, -1, -1> output(targetHeight * targetWidth, imgChannel * kernel_w * kernel_h);
  
  Dtype* data = src.data() + imgWidth * imgHeight * imgChannel * numIndex;
  
  for (int c = 0; c < imgChannel * kernel_w * kernel_h; c++) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_w / kernel_h;
    
    for (int h = 0; h < targetHeight; ++h) {
      for (int w = 0; w < targetWidth; ++w) {
        int im_row = h_offset + h * stride_h;
        int im_col = w_offset + w * stride_w;
        
        //int col_index = (c * targetHeight + h) * targetWidth + w;
        im_row -= pad_h;
        im_col -= pad_w;
        
        Dtype value = 0;
        if ( im_row >= 0 && im_row < imgHeight && im_col >=0 && im_col < imgWidth) {
          value = data[c_im * imgWidth * imgHeight + im_row * imgWidth + im_col];
        }
        output(h*targetWidth + w, c) = value;
      }
    }
  }
  
  return output;
}

template <typename Dtype>
Eigen::Matrix<Dtype, -1, -1> im2col<Dtype>::filter2col_cpu(Blob<Dtype>& filter, const int number,
                                            const int targetHeight, const int targetWidth) {
  assert(filter.device() == -1);
  assert(number >= 1);
  
  int col_height = targetWidth * targetHeight;
  int col_width = number * filter.size();
  
  Dtype* data = filter.data();
  int filterSize = filter.size();
  Eigen::Matrix<Dtype, -1, -1> output(col_height, col_width);
  
  for (int i = 0; i < col_height; i++) {
    for (int j = 0; j < col_width; j++) {
      output(i,j) = data[j % filterSize];
    }
  }
  
  return output;
}
  
INSTANTIATE_CLASS(im2col);
}
