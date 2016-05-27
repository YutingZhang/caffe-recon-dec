#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/depooling_layer.hpp"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void DepoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	pooling_->Backward_gpu( pool_top_, backward_tag4forward_, pool_bottom_ );

	if (top.size()>1) {
		Dtype* top_weights = top[1]->mutable_gpu_data();
		const Dtype* pool_bweights = pooling_->backward_weights()->gpu_data();
		caffe_copy( top[1]->count(), pool_bweights, top_weights );
	}

}


template <typename Dtype>
void DepoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}

	pooling_->Forward_gpu( pool_bottom_, pool_top_ );

}


INSTANTIATE_LAYER_GPU_FUNCS(DepoolingLayer);


}  // namespace caffe
