#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask, bool use_local_idx) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart0 = ph * stride_h - pad_h;
    int wstart0 = pw * stride_w - pad_w;
    const int hend = min(hstart0 + kernel_h, height);
    const int wend = min(wstart0 + kernel_w, width);
    int hstart = max(hstart0, 0);
    int wstart = max(wstart0, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;

    int stored_maxidx = use_local_idx ?
  		  ( (maxidx/width-hstart0)*kernel_w+(maxidx%width-wstart0) ):(maxidx);
    if (mask) {
      mask[index] = stored_maxidx;
    } else {
      top_mask[index] = static_cast<Dtype>( stored_maxidx );
    }

  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__
void SumPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype sumval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        sumval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = sumval;
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
__global__ void FixPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int fix_x, const int fix_y,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int h = hstart + fix_y;
    int w = wstart + fix_x;
    if (h<0 || h>=height || w<0 || w>=width) {
    	top_data[index] = Dtype(0);
    } else {
        const Dtype* const bottom_slice =
            bottom_data + (n * channels + c) * height * width;
    	top_data[index] = bottom_slice[h * width + w];
    }

  }
}

template <typename Dtype>
__global__ void SwitchPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const Dtype* bottom_switch,
    Dtype* const top_data, bool use_local_idx) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    if (use_local_idx) {
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		const int local_index = bottom_switch[index];
		int h = hstart+local_index/kernel_w;
		int w = wstart+local_index%kernel_w;
		if (h<0 || h>=height || w<0 || w>=width) {
			top_data[index] = Dtype(0);
		} else {
			const Dtype* const bottom_slice =
				bottom_data + (n * channels + c) * height * width;
			top_data[index] = bottom_slice[h * width + w];
		}
    } else {
	    const Dtype* const bottom_slice =
            bottom_data + (n * channels + c) * height * width;
    	const int bottom_index = bottom_switch[index];
    	top_data[index] = bottom_slice[ bottom_index ];
    }

  }
}


template <typename Dtype>
__global__
void SoftSwitchPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const Dtype* bottom_switch, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart0 = ph * stride_h - pad_h;
    int wstart0 = pw * stride_w - pad_w;
    int hend = min(hstart0 + kernel_h, height + pad_h);
    int wend = min(wstart0 + kernel_w, width + pad_w);
    int hstart = max(hstart0, 0);
    int wstart = max(wstart0, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;

    const int pooled_geo_count = pooled_width*pooled_height;
    const Dtype* switch_slice = bottom_switch +  (n*channels+c)*kernel_w*kernel_h*pooled_geo_count;
    const int slided_index = ph*pooled_width+pw;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
    	  int local_index  = (h-hstart0)*kernel_w+(w-wstart0);
    	  int switch_index = local_index*pooled_geo_count + slided_index;
          aveval += bottom_slice[h * width + w] * switch_slice[switch_index];
      }
    }
    top_data[index] = aveval;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask, mask_index_type_ == PoolingParameter_MaskIndexType_LOCAL );
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SumPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  case PoolingParameter_PoolMethod_FIX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    FixPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        fix_x_, fix_y_,
        top_data);
    break;
  case PoolingParameter_PoolMethod_SWITCH:
    // NOLINT_NEXT_LINE(whitespace/operators)
  {
	const Dtype* bottom_switch = bottom[1]->gpu_data();
    SwitchPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_switch, top_data, mask_index_type_ == PoolingParameter_MaskIndexType_LOCAL);
  }
    break;
  case PoolingParameter_PoolMethod_SOFT_SWITCH:
    // NOLINT_NEXT_LINE(whitespace/operators)
  {
	const Dtype* bottom_switch = bottom[1]->gpu_data();
    SoftSwitchPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_switch, top_data);
  }
    break;

  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff, Dtype* const bweights, bool use_local_idx ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0; Dtype bwgt = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (use_local_idx) {
		if (mask) {
		  const int* const mask_slice = mask + offset;
		  for (int ph = phstart; ph < phend; ++ph) {
	    	int hstart = ph * stride_h - pad_h;
			for (int pw = pwstart; pw < pwend; ++pw) {
	    	  int wstart = pw * stride_w - pad_w;
	    	  int local_index = (h-hstart)*kernel_w+(w-wstart);
			  if (mask_slice[ph * pooled_width + pw] == local_index) {
				gradient += top_diff_slice[ph * pooled_width + pw];
				bwgt += Dtype(1.);
			  }
			}
		  }
		} else {
		  const Dtype* const top_mask_slice = top_mask + offset;
		  for (int ph = phstart; ph < phend; ++ph) {
    		int hstart = ph * stride_h - pad_h;
			for (int pw = pwstart; pw < pwend; ++pw) {
    		  int wstart = pw * stride_w - pad_w;
    		  int local_index = (h-hstart)*kernel_w+(w-wstart);
			  if (top_mask_slice[ph * pooled_width + pw] == local_index) {
				gradient += top_diff_slice[ph * pooled_width + pw];
				bwgt += Dtype(1.);
			  }
			}
		  }
		}
    } else {
		if (mask) {
		  const int* const mask_slice = mask + offset;
		  for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
			  if (mask_slice[ph * pooled_width + pw] == h * width + w) {
				gradient += top_diff_slice[ph * pooled_width + pw];
				bwgt += Dtype(1.);
			  }
			}
		  }
		} else {
		  const Dtype* const top_mask_slice = top_mask + offset;
		  for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
			  if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
				gradient += top_diff_slice[ph * pooled_width + pw];
				bwgt += Dtype(1.);
			  }
			}
		  }
		}
    }
    bottom_diff[index] = gradient;
    bweights[index] = bwgt;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff, Dtype* const bweights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0; Dtype bwgt = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
        bwgt += Dtype(1.)/Dtype(pool_size);
      }
    }
    bottom_diff[index] = gradient;
    bweights[index] = bwgt;
  }
}

template <typename Dtype>
__global__
void SumPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff, Dtype* const bweights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0; Dtype bwgt = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        gradient += top_diff_slice[ph * pooled_width + pw];
        bwgt += Dtype(1.);
      }
    }
    bottom_diff[index] = gradient;
    bweights[index] = bwgt;
  }
}



template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff, Dtype* const bweights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0; Dtype bwgt = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
        bwgt += (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
    bweights[index] = bwgt;
  }
}


template <typename Dtype>
__global__ void FixPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int fix_x, const int fix_y,
    Dtype* const bottom_diff, Dtype* const bweights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	// index is the pooled index (it is safe to do it here)
	const int pw = index % pooled_width;
	const int ph = (index / pooled_width) % pooled_height;
	const int c = (index / pooled_width / pooled_height) % channels;
	const int n = index / pooled_width / pooled_height / channels;
	int hstart = ph * stride_h - pad_h;
	int wstart = pw * stride_w - pad_w;
	int h = hstart + fix_y;
	int w = wstart + fix_x;
	if (!(h<0 || h>=height || w<0 || w>=width)) {
    	const int depooled_index = (n * channels + c) * height * width + h * width + w;
		bottom_diff[depooled_index] = top_diff[index];
		bweights[depooled_index] = Dtype(1.);
	}
  }
}


template <typename Dtype>
__global__
void SoftSwitchPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const Dtype* const bottom_switch, Dtype* const bottom_diff, Dtype* const bweights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0; Dtype bwgt = 0;

    const int pooled_geo_count = pooled_height * pooled_width;
    const int ch_count = n * channels + c;
    const int kernel_count = kernel_w * kernel_h;

    const Dtype* const top_diff_slice =
        top_diff + ch_count * pooled_geo_count;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;

        int local_index = (h-hstart)*kernel_w+(w-wstart);
        int top_sliced_index = ph * pooled_width + pw;
        int switch_index =  (ch_count * kernel_count + local_index) * pooled_geo_count + top_sliced_index;

        gradient += top_diff_slice[top_sliced_index] * bottom_switch[switch_index];
        bwgt += bottom_switch[switch_index];
      }
    }
    bottom_diff[index] = gradient;
    bweights[index] = bwgt;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  Dtype* bweights = backward_weights_.mutable_gpu_data();

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_SWITCH:
	top_mask = bottom[1]->gpu_data();
  case PoolingParameter_PoolMethod_MAX:
	if (!top_mask) {
		if (use_top_mask) {
		  top_mask = top[1]->gpu_data();
		} else {
		  mask = max_idx_.gpu_data();
		}
	}
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff, bweights, mask_index_type_ == PoolingParameter_MaskIndexType_LOCAL);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff, bweights);
    break;
  case PoolingParameter_PoolMethod_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SumPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff, bweights);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff, bweights);
    break;
  case PoolingParameter_PoolMethod_FIX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    {
	const int top_count = top[0]->count();
	caffe_gpu_set(count, Dtype(0.), bottom_diff);
	caffe_gpu_set(count, Dtype(0.), bweights);
    FixPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        top_count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        fix_x_, fix_y_,
        bottom_diff, bweights);
    }
    break;
  case PoolingParameter_PoolMethod_SOFT_SWITCH:
    // NOLINT_NEXT_LINE(whitespace/operators)
  {
	const Dtype* bottom_switch = bottom[1]->gpu_data();
    SoftSwitchPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_switch, bottom_diff, bweights);
  }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
