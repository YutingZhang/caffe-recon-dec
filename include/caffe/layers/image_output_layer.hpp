#ifndef CAFFE_IMAGE_OUTPUT_LAYER_HPP_
#define CAFFE_IMAGE_OUTPUT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <fstream>
#include <vector>

namespace caffe {

template <typename Dtype>
class ImageOutputLayer : public Layer<Dtype> {
 public:
  explicit ImageOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param), iter_(0), 
      image_idx_(0), filename_padding_(10), print_iter_(0),
      norm_method_(ImageOutputParameter_NormalizationMethod_SCALE), 
      scale_(1.), offset_(0) {}
  virtual ~ImageOutputLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageOutput"; }
  // TODO: no limit on the number of blobs
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

  inline std::string file_name() const { return file_prefix_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  size_t iter_;
  std::string file_prefix_;
  int image_idx_;
  int filename_padding_;
  int print_iter_;
  int norm_method_;
  Dtype scale_;
  Dtype offset_;
  Blob<Dtype> buf_;
  bool force_slice_channels_;
  Dtype auto_ratio_;
};

}

#endif

