#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/depooling_layer.hpp"

#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

using std::min;
using std::max;


LayerParameter DepoolingLayer_PoolLayer_Param( const LayerParameter& p ) {
    LayerParameter q;
    q.set_name( p.name() + " [reverse pooling]" );
    q.set_type( "Pooling" );
    auto& pool_param = *(q.mutable_pooling_param());
    pool_param = p.pooling_param();
    if (pool_param.pool()==PoolingParameter_PoolMethod_MAX) {
    	pool_param.set_pool(PoolingParameter_PoolMethod_SWITCH);
    }
    return q;
}

template <typename Dtype>
DepoolingLayer<Dtype>::DepoolingLayer(const LayerParameter& param) :
		Layer<Dtype>(param), 
        pooling_( new PoolingLayer<Dtype>( DepoolingLayer_PoolLayer_Param(param) ) ), 
        pool_top_(1, &pool_top_0_), pool_bottom_(
				1, &pool_bottom_0_) {

	*(this->layer_param_.mutable_pooling_param()) = 
        pooling_->layer_param_.pooling_param();

	int pool_type = this->layer_param_.pooling_param().pool();
	if (pool_type == PoolingParameter_PoolMethod_SWITCH ||
			pool_type == PoolingParameter_PoolMethod_SOFT_SWITCH ) {
		pool_bottom_.push_back(&max_mask_);
	}

	pooling_->memory_op_ = false;
    
}

template <typename Dtype>
void DepoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

    PoolingParameter pool_param = this->layer_param_.pooling_param();

	if (this->layer_param_.pooling_param().pool() ==
			PoolingParameter_PoolMethod_STOCHASTIC ) {
		NOT_IMPLEMENTED;
	}

	CHECK( !pool_param.global_pooling() ) << "Depooling does not support global_pooling, as it cannot figure out the original blob shape";


	// pool_top_[0]->ShareSwappedDataDiff(*bottom[0]);
	// pool_bottom_[0]->ShareSwappedDataDiff(*top[0]);

	pooling_->LayerSetUp( pool_bottom_, pool_top_ );

    
    backward_tag4forward_.resize(1,true);
    backward_tag4forward_[0] = true;

}

template <typename Dtype>
void DepoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	// Reshape manually
	int num_, channels_, height_, width_, depooled_height_, depooled_width_;
	num_      = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_   = bottom[0]->height();
	width_    = bottom[0]->width();
	depooled_height_ = static_cast<int>(static_cast<float>(height_ - 1) * pooling_->stride_h_
			+ pooling_->kernel_h_ - 2 * pooling_->pad_h_);
	depooled_width_ = static_cast<int>(static_cast<float>(width_ - 1) * pooling_->stride_w_
			+ pooling_->kernel_w_ - 2 * pooling_->pad_w_);

	top[0]->Reshape(num_, channels_, depooled_height_, depooled_width_);
	pool_bottom_[0]->ReshapeLike(*top[0]);
	pool_bottom_[0]->ShareSwappedDataDiff(*top[0]);

	pool_top_[0]->ReshapeLike(*bottom[0]);
	pool_top_[0]->ShareSwappedDataDiff(*bottom[0]);

	int pool_type = this->layer_param_.pooling_param().pool();
	if (pool_type == PoolingParameter_PoolMethod_SWITCH ||
			pool_type == PoolingParameter_PoolMethod_SOFT_SWITCH ) {
	    pool_bottom_[1]->ReshapeLike(*bottom[1]);
		pool_bottom_[1]->ShareData(*bottom[1]);
		pool_bottom_[1]->ShareDiff(*bottom[1]);
	}

	pooling_->Reshape( pool_bottom_, pool_top_ );

	CHECK_EQ( height_, pooling_->pooled_height_) << "mismatched shape between pooling and depooling";
	CHECK_EQ( width_,  pooling_->pooled_width_ ) << "mismatched shape between pooling and depooling";

    if (top.size()>1)
		top[1]->ReshapeLike( *(pooling_->backward_weights()) );

}

template <typename Dtype>
void DepoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	pooling_->Backward_cpu( pool_top_, backward_tag4forward_, pool_bottom_ );

	if (top.size()>1) {
		Dtype* top_weights = top[1]->mutable_cpu_data();
		const Dtype* pool_bweights = pooling_->backward_weights()->cpu_data();
		caffe_copy( top[1]->count(), pool_bweights, top_weights );
	}

}

template <typename Dtype>
void DepoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (!propagate_down[0]) return;

    pooling_->Forward_cpu( pool_bottom_, pool_top_ );

}


#ifdef CPU_ONLY
STUB_GPU(DepoolingLayer);
#endif

INSTANTIATE_CLASS(DepoolingLayer);
REGISTER_LAYER_CLASS(Depooling);

}  // namespace caffe
