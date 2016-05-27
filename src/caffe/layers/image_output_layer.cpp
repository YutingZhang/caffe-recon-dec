// Yuting Zhang
#ifdef USE_OPENCV

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/image_output_layer.hpp"
#include <boost/algorithm/minmax_element.hpp>
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <utility>
#include <sstream>
#include <iomanip>

namespace caffe {


template <typename Dtype>
void ImageOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	const ImageOutputParameter& imout_param = this->layer_param_.image_output_param();
    iter_ = 0;
	file_prefix_ = imout_param.output_prefix();
	image_idx_   = imout_param.base_idx();
    filename_padding_ = imout_param.filename_padding();
    print_iter_  = imout_param.print_iter();
	norm_method_ = imout_param.normalization();
	force_slice_channels_ = imout_param.force_slice_channels();
	switch ( norm_method_ ) {
	case ImageOutputParameter_NormalizationMethod_SCALE:
		scale_  = imout_param.scale();
		offset_ = imout_param.offset();
		break;
	case ImageOutputParameter_NormalizationMethod_AutoSCALE:
		auto_ratio_ = imout_param.auto_ratio();
		CHECK(auto_ratio_>0. && auto_ratio_<=1.) << "auto_ratio must be >0 and <=1";
		break;
	default:
		LOG(ERROR) << "Unknown normalization method";
	}
}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Set up the cache for re-scaling
  buf_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ImageOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    ++iter_;
	Dtype* buf = buf_.mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num(), s = count/num;
	switch ( norm_method_ ) {
	case ImageOutputParameter_NormalizationMethod_SCALE:
		caffe_cpu_scale( count, scale_, bottom_data, buf );
		caffe_add_scalar( count, offset_, buf );
		break;
	case ImageOutputParameter_NormalizationMethod_AutoSCALE:
	{
		int num_removed = s-static_cast<int>(std::ceil(Dtype(s)*auto_ratio_));
		for ( int i=0 ; i<num ; ++i ) {
			const Dtype* p = bottom_data + i*s;
			Dtype* q = buf+i*s;
			Dtype minV, maxV;
			if (num_removed) {
				caffe_copy( s, p, q );
				std::sort(q,q+s);
				Dtype* e1 = q, * e2 = q+s-1;
				for ( int j=0; j<num_removed; ++j ) {
					if (*(e1+1)-*(e1) > *(e2)-*(e2-1)) {
						++e1;
					} else {
						--e2;
					}
				}
				minV = *e1; maxV = *e2;
			} else {
				std::pair< const Dtype*, const Dtype* > mm =
						boost::minmax_element(p, p+s);
				minV = *(mm.first), maxV = *(mm.second);
			}
            //LOG(INFO) << "min, max : " << minV << "\t" << maxV;
			Dtype scaleV = maxV-minV;
			if (scaleV<=0) scaleV=Dtype(1.);
			caffe_cpu_scale( s, 1/scaleV, p, q );
			caffe_add_scalar( s, -minV/scaleV, q );
		}
		break;
	}
	default:
		LOG(ERROR) << "Internal: unknown normalization method";
	}
    std::string cur_prefix;
    if (print_iter_>0) {
	    std::ostringstream ss;
		ss << std::setw(print_iter_) << std::setfill('0') << iter_;
        cur_prefix = file_prefix_ + ss.str() + "-";
    } else {
        cur_prefix = file_prefix_;
    }
	image_idx_ = SaveArrayAsImages( bottom[0]->width(), bottom[0]->height(),
			bottom[0]->channels(), bottom[0]->num(), buf,
			cur_prefix, image_idx_, filename_padding_, force_slice_channels_ );
}

#ifdef CPU_ONLY
STUB_GPU(ImageOutputLayer);
#endif

INSTANTIATE_CLASS(ImageOutputLayer);
REGISTER_LAYER_CLASS(ImageOutput);

}  // namespace caffe

#endif
