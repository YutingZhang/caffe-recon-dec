/*
 * safe_inv_layer.cpp
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
void SafeInvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  zcaffe_cpu_safeinv<Dtype>( count, bottom_data, top_data );
}

template <typename Dtype>
void SafeInvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    /*
    caffe_mul(count, top_data, top_data, bottom_diff);
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    caffe_scal(count, Dtype(-1.), bottom_diff);
    */

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = -top_diff[i] * top_data[i] * top_data[i];
    }

  }
}

#ifdef CPU_ONLY
STUB_GPU(SafeInvLayer);
#endif

INSTANTIATE_CLASS(SafeInvLayer);
REGISTER_LAYER_CLASS(SafeInv);


}  // namespace caffe
