/*
 * nan2zero_layer.cpp
 *
 *  Created on: June 19, 2015
 *      Author: zhangyuting
 */


#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/safe_inv_layer.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SafeInvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  zcaffe_gpu_safeinv<Dtype>(count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SafeInvBackward(const int n, const Dtype* top_diff,
    const Dtype* top_data, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    bottom_diff[index] = - top_diff[index] * top_data[index] * top_data[index];
  }
}

template <typename Dtype>
void SafeInvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SafeInvBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SafeInvLayer);

}  // namespace caffe

