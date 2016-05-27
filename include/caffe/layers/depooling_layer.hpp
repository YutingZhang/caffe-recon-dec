#ifndef CAFFE_DEPOOLING_LAYER_HPP_
#define CAFFE_DEPOOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/* Depooling layer
 * A more decent version of Unpooling
 */

template <typename Dtype> class PoolingLayer;

template <typename Dtype>
class DepoolingLayer : public Layer<Dtype> {
protected:
	shared_ptr< PoolingLayer<Dtype> > pooling_;
	Blob<Dtype> pool_top_0_, pool_bottom_0_, max_mask_;
	vector<Blob<Dtype>*> pool_top_, pool_bottom_;
public:
 explicit DepoolingLayer(const LayerParameter& param);
 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

 virtual inline const char*  type() const { return "Depooling"; }
 virtual inline int MinNumTopBlobs() const { return 1; }
 virtual inline int MaxNumTopBlobs() const { return 2; }
 virtual inline int ExactBottomBlobs() const {
	 int pooling_type = this->layer_param_.pooling_param().pool();
	 return ( pooling_type == PoolingParameter_PoolMethod_MAX ||
			 pooling_type  == PoolingParameter_PoolMethod_SWITCH ||
			 pooling_type  == PoolingParameter_PoolMethod_SOFT_SWITCH ) ? 2 : 1;
 }

protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 vector<bool> backward_tag4forward_;

};

}

#endif

