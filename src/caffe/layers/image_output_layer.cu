// Yuting Zhang

#ifdef USE_OPENCV

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/image_output_layer.hpp"

namespace caffe {


template <typename Dtype>
void ImageOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	this->Forward_cpu( bottom, top );
}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { }


INSTANTIATE_LAYER_GPU_FUNCS(ImageOutputLayer);

}  // namespace caffe

#endif
