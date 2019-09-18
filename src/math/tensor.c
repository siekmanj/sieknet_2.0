#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <tensor.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Returns a number from uniform distribution bounded
 * by the given parmeters.
 */
float uniform(float lowerbound, float upperbound){
	return lowerbound + (upperbound - lowerbound) * ((float)rand()/RAND_MAX);
}

/*
 * Returns a number from a normal distribution defined
 * by the given parameters using the Box-Muller transform.
 */
float normal(float mean, float std){
	float u1 = uniform(1e-6, 1);
	float u2 = uniform(1e-6, 1);
	float norm = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
	return mean + norm * std;
}

/*
 * Returns an int if two tensors have the same dimensions
 */
int tensor_dimensions_match(Tensor a, Tensor b){
  if(a.n != b.n)
    return 0;

  for(int i = 0; i < a.n; i++)
    if(a.dims[i] != b.dims[i])
      return 0;

  return 1;
}

/* 
 * Returns the raw pointer to memory contained by tensor.
 */
float *tensor_raw(Tensor t){
#ifdef SIEKNET_DEBUG
  if(t.device != SIEKNET_CPU)
    SK_ERROR("Cannot provide memory address of tensor not on CPU.");
#endif
  return &((float*)t.data)[t.data_offset];
}

/*
 * Returns the argmax of a 1d tensor.
 */
int tensor_argmax(Tensor t){
  if(t.n != 1)
    SK_ERROR("can only get argmax of vectors");

  int argmax = 0;
  for(int i = 1; i < t.size; i++){
    float curr = tensor_at(t, i);
    if(curr > tensor_at(t, argmax))
      argmax = i;
  }
  return argmax;
}

/*
 * Computes the index offset of a location in the tensor's
 * memory using 'arr' as an array of indices.
 */
size_t tensor_flat_idx(Tensor t, size_t *arr, size_t len){
  size_t idx = 0;

  size_t max_idx = MIN(len, t.n);
  for(int i = 0; i < max_idx; i++)
    idx += t.strides[i] * arr[i];

  return idx;
}

/*
 * Returns the element at the provided index of a tensor.
 */
float tensor_at_idx(Tensor t, size_t *arr, size_t len){
  if(t.device == SIEKNET_CPU){
    return tensor_raw(t)[tensor_flat_idx(t, arr, len)];

  }else if(t.device == SIEKNET_GPU){

  }
  return -1;
}

/*
 * Returns the cosine similarity of two tensors.
 * Reductions are cheaper on the CPU probably.
 */
float tensor_cosine_similarity(Tensor a, Tensor b){
  if(!tensor_dimensions_match(a, b))
    SK_ERROR("Tensor dims must match!");
 
  if(a.n != 1 || b.n != 1)
    SK_ERROR("Can only do cosine similarity of two vectors.");

  float dot_product = 0;
  float a_mag = 0;
  float b_mag = 0;
  for(int i = 0; i < a.size; i++){
    float a_i = tensor_at(a, i);
    float b_i = tensor_at(b, i);
    dot_product += a_i * b_i;
    a_mag += a_i * a_i;
    b_mag += b_i * b_i;
  }
  return dot_product / sqrt(a_mag * b_mag);
}

/*
 * Fills the tensor's memory with random (normally distributed)
 * numbers.
 */
void tensor_fill_random(Tensor t, float mean, float std){
  size_t pos[t.n];
  memset(pos, '\0', sizeof(size_t)*t.n);

  for(int i = 0; i < t.size; i++){
    tensor_raw(t)[tensor_flat_idx(t, pos, t.n)] = normal(mean, std);
    pos[t.n - 1]++;
    for(int j = t.n - 1; j > 0; j--){
      if(!(pos[j] % t.dims[j])){
        pos[j-1]++;
        pos[j] = 0;
      }else break;
    }
  }
}

/*
 * Fills the tensor's memory with the specified value.
 */
void tensor_fill(Tensor t, float val){
  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size; i++){
      tensor_raw(t)[tensor_flat_idx(t, pos, t.n)] = val;
      pos[t.n - 1]++;
      for(int j = t.n - 1; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else
    SK_ERROR("Not supported.");
}

/*
 * Copies a tensor's dimensions and stride layout, but not
 * its memory.
 */
Tensor tensor_clone(TENSOR_DEVICE device, Tensor src){
  Tensor ret = {0};
  ret.n = src.n;
  ret.dims = malloc(ret.n * sizeof(size_t));
  ret.strides = malloc(ret.n * sizeof(size_t));
  ret.size = src.size;
  ret.type = TENSOR;
  ret.data_offset = 0;
  memcpy(ret.dims, src.dims, src.n * sizeof(size_t));
  memcpy(ret.strides, src.strides, src.n * sizeof(size_t));

  if(device == SIEKNET_CPU)
    ret.data = calloc(ret.size, sizeof(float));
  else
    SK_ERROR("Not implemented.");

  return ret;
}

/*
 * Copies the contents of a tensor into another tensor.
 */
void tensor_copy(Tensor src, Tensor dest){

  if(src.device != dest.device)
    SK_ERROR("Tensors must be on the same device.");

  if(src.n != dest.n)
    SK_ERROR("Tensors must have same number of dimensions.");

  for(int i = 0; i < src.n; i++)
    if(src.dims[i] != dest.dims[i])
      SK_ERROR("Tensor dimensions must match! Mismatch on dimension %d: %lu vs %lu\n", i, src.dims[i], dest.dims[i]);
  
  size_t len = 1;
  for(int i = 0; i < src.n; i++)
    len *= src.dims[i];

  size_t pos[src.n];
  memset(pos, '\0', sizeof(size_t)*src.n);

  if(src.device == SIEKNET_CPU){
    for(int i = 0; i < src.size; i++){
      tensor_raw(dest)[tensor_flat_idx(dest, pos, dest.n)] = tensor_raw(src)[tensor_flat_idx(src, pos, src.n)];
      pos[src.n - 1]++;
      for(int j = src.n - 1; j > 0; j--){
        if(!(pos[j] % src.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else{
    SK_ERROR("Not implemented.");
  }
}

/*
 * Performs the sigmoid logistic function on a tensor. Also
 * computes the intermediate gradient (derivative of sigmoid).
 */
void tensor_sigmoid_precompute(Tensor t, Tensor d){
  if(t.n > 1)
    SK_ERROR("Logistics not supported for non-1d tensors.");

  if(d.data != NULL && t.n != d.n)
    SK_ERROR("If derivative tensor is supplied, dimensions must match. T dims: %lu, d dims: %lu", t.n, d.n);

  for(int i = 0; i < t.n && d.data != NULL; i++)
    if(t.dims[i] != d.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu\n", i, t.dims[i], d.dims[i]);

  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size; i++){
      float *t_raw = &tensor_raw(t)[tensor_flat_idx(t, pos, t.n)];
      float *d_raw = &tensor_raw(d)[tensor_flat_idx(d, pos, d.n)];
      *t_raw = 1 / (1 + exp(-*t_raw));
      *d_raw = *t_raw * (1 - *t_raw);

      pos[t.n - 1]++;
      for(int j = t.n - 1; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else
    SK_ERROR("Not implemented.");
}

/*
 * Performs the tanh logistic function on a tensor. Also
 * computes the intermediate gradient (derivative of tanh).
 */
void tensor_tanh_precompute(Tensor t, Tensor d){
  if(d.data != NULL && t.n != d.n)
    SK_ERROR("If derivative tensor is supplied, dimensions must match. T dims: %lu, d dims: %lu", t.n, d.n);

  for(int i = 0; i < t.n && d.data != NULL; i++)
    if(t.dims[i] != d.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu\n", i, t.dims[i], d.dims[i]);

  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size; i++){
      float *t_raw = &tensor_raw(t)[tensor_flat_idx(t, pos, t.n)];
      float *d_raw = &tensor_raw(d)[tensor_flat_idx(d, pos, d.n)];
      *t_raw = (exp(*t_raw) - exp(-*t_raw)) / (exp(*t_raw) + exp(-*t_raw));
      *d_raw = 1 - (*t_raw * *t_raw);

      pos[t.n - 1]++;
      for(int j = t.n - 1; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else
    SK_ERROR("Not implemented.");
}

/*
 * Performs the relu nonlinearity on a tensor. Also
 * computes the intermediate gradient (derivative of relu)
 */
void tensor_relu_precompute(Tensor t, Tensor d){
  if(d.data != NULL && t.n != d.n)
    SK_ERROR("If derivative tensor is supplied, dimensions must match. T dims: %lu, d dims: %lu", t.n, d.n);

  for(int i = 0; i < t.n && d.data != NULL; i++)
    if(t.dims[i] != d.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu\n", i, t.dims[i], d.dims[i]);

  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size; i++){
      float *t_raw = &tensor_raw(t)[tensor_flat_idx(t, pos, t.n)];
      float *d_raw = &tensor_raw(d)[tensor_flat_idx(d, pos, d.n)];
      *t_raw = *t_raw > 0 ? *t_raw : 0;
      *d_raw = *t_raw > 0 ? 1 : 0;

      pos[t.n - 1]++;
      for(int j = t.n - 1; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else
    SK_ERROR("Not implemented.");
}

/*
 * Performs a linear transform on a tensor.
 * (i.e., doesn't do anything)
 */
void tensor_linear_precompute(Tensor t, Tensor d){
  tensor_fill(d, 1.0f);
}

/*
 * Performs the softmax nonlinearity on a tensor.
 */
void tensor_softmax_precompute(Tensor t, Tensor d){
  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size/t.dims[t.n-1]; i++){
      Tensor vec = tensor_to_subtensor(t, pos, t.n-1);

      // FUTURE:
      // tensor_exp(vec);
      // float sum = tensor_reduce(vec);
      // tensor_elementwise_mul(vec, 1/sum);
      // easy peasy, works on cpu/gpu

			double arr_max = 0;
      for(int j = 0; j < vec.size; j++){
        if(tensor_at(vec, j) > arr_max)
					arr_max = tensor_at(vec, j);
      }

      double sum = 0;
      for(int j = 0; j < vec.size; j++){
        sum += exp(tensor_at(vec, j) - arr_max);
      }

      for(int j = 0; j < vec.size; j++){
        tensor_raw(vec)[tensor_get_offset(vec, j)] = (double)exp(tensor_at(vec, j) - arr_max) / (double)sum;
      }

      pos[t.n - 2]++;
      for(int j = t.n - 2; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
    
    if(d.data){
      memset(pos, '\0', sizeof(size_t)*t.n);
      tensor_fill(d, 0.0f);

      for(int i = 0; i < t.size/t.dims[t.n-1]; i++){
        Tensor vec = tensor_to_subtensor(t, pos, t.n-1);
        Tensor jac = tensor_to_subtensor(d, pos, d.n-2);

				if(jac.size != vec.size * vec.size)
					SK_ERROR("Expected jacobian to be vec^2 - vec was %lu and jacobian was %lu x %lu", vec.size, jac.dims[0], jac.n > 1 ? jac.dims[1] : 1);

        for(int j = 0; j < vec.size; j++){
          for(int k = 0; k < vec.size; k++){
            float s_j = tensor_at(vec, j);
            float s_k = tensor_at(vec, k);
            tensor_raw(jac)[tensor_get_offset(jac, j, k)] = s_k * ((j == k) - s_j);
          }
        }
        pos[t.n - 2]++;
        for(int j = t.n - 2; j > 0; j--){
          if(!(pos[j] % t.dims[j])){
            pos[j-1]++;
            pos[j] = 0;
          }else break;
        }
      }
    }
  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

/*
 * Returns the euclidean cost given two column vectors/tensors,
 * and stores the gradient of the cost function in a third tensor.
 */
double tensor_quadratic_cost(Tensor o, Tensor y, Tensor grad){
  if(y.device != o.device || y.device != grad.device)
    SK_ERROR("Devices must match.");

  if(y.n != 1 || o.n != 1 || grad.n != 1)
    SK_ERROR("All tensor dimensions must be 1. Got %lu, %lu, %lu.", y.n, o.n, grad.n);

  if(y.dims[0] != o.dims[0] || y.dims[0] != grad.dims[0])
    SK_ERROR("All tensors must be of the same length, got lengths %lu, %lu, %lu", y.dims[0], o.dims[0], grad.dims[0]);

  double cost = 0;
  if(y.device == SIEKNET_CPU){

    float *o_mem = tensor_raw(o);
    float *y_mem = tensor_raw(y);
    float *g_mem = tensor_raw(grad);
    for(int i = 0; i < y.dims[0]; i++){
      float o_i = o_mem[i * o.strides[0]];
      float y_i = y_mem[i * y.strides[0]];

      cost += 0.5 * (o_i - y_i) * (o_i - y_i);
      g_mem[i * grad.strides[0]] = (o_i - y_i);
    }

    return cost;
  }else if(y.device == SIEKNET_GPU){
    SK_ERROR("Tensor cost not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

/*
 * Returns the cross-entropy cost given two column vectors/tensors,
 * and stores the gradient of the cost function in a third tensor.
 */
double tensor_cross_entropy_cost(Tensor o, Tensor y, Tensor grad){
  if(y.device != o.device || y.device != grad.device)
    SK_ERROR("Devices must match.");

  if(y.n != 1 || o.n != 1 || grad.n != 1)
    SK_ERROR("All tensor dimensions must be 1. Got %lu, %lu, %lu.", y.n, o.n, grad.n);

  if(y.dims[0] != o.dims[0] || y.dims[0] != grad.dims[0])
    SK_ERROR("All tensors must be of the same length, got lengths %lu, %lu, %lu", y.dims[0], o.dims[0], grad.dims[0]);

  double cost = 0;
  if(y.device == SIEKNET_CPU){
    float *o_mem = tensor_raw(o);
    float *y_mem = tensor_raw(y);
    float *g_mem = tensor_raw(grad);
    for(int i = 0; i < y.dims[0]; i++){
      float o_i = o_mem[i * o.strides[0]];
      float y_i = y_mem[i * y.strides[0]];

      o_i = MAX(o_i, 1e-3);
      y_i = MIN(y_i, 0.999);

      cost += - y_i * log(o_i);
      g_mem[i * grad.strides[0]] = (o_i - y_i) / (o_i * (1 - o_i));
    }
    return cost;
  }else if(y.device == SIEKNET_GPU){
    SK_ERROR("Tensor cost not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

/*
 * Performs an O(1) transpose on a tensor by flipping the strides.
 * Does not rearrange memory in any way - may lead to unexpected
 * slowdowns as a result.
 */
void tensor_transpose(Tensor t, size_t dim1, size_t dim2){
  if(dim1 > t.n || dim2 > t.n)
    SK_ERROR("Invalid axes (%lu, %lu) for tensor with dimension %lu.", dim1, dim2, t.n);

  SWAP(t.dims[t.n - dim1 - 1], t.dims[t.n - dim2 - 1]);
  SWAP(t.strides[t.n - dim1 - 1], t.strides[t.n - dim2 - 1]);

}

/*
 * Calculates the stride for iterating over a given axis.
 */
static size_t tensor_axis_stride(Tensor t, size_t axis){
  size_t stride = 1;
  for(int i = 0; i < axis; i++)
    stride *= t.dims[t.n - i - 1];
  return stride;
}


/*
 * Retrieves a subtensor from a tensor. Does not allocate new 
 * memory - just recalculates offsets of original memory.
 */
static size_t tmp_dim = 1;
static size_t tmp_str = 0;
Tensor tensor_to_subtensor(Tensor t, size_t *arr, size_t len){
  Tensor ret = t;

  if(len > t.n){
    SK_ERROR("Cannot get subtensor with dimension less than 1 from tensor of dimension %lu\n", t.n);
  }
  else if(len == t.n){
    ret.n            = 1;
    ret.dims         = &tmp_dim;
    ret.strides      = &tmp_str;
    ret.data_offset  += tensor_flat_idx(t, arr, len);
  }else{
    ret.n            = t.n - len;
    ret.dims         = &t.dims[len];
    ret.strides      = &t.strides[len];
    ret.data_offset  += tensor_flat_idx(t, arr, len);
  }
  ret.type = SUBTENSOR;
  ret.size = ret.strides[0] * ret.dims[0];

  return ret;
}

/*
 * Retrieves a subtensor from a tensor, creating a new shape in
 * the process. Allocates new memory for dimensions and strides,
 * does not allocate new memory for the tensor data itself.
 */
Tensor tensor_to_subtensor_reshape(Tensor t, size_t offset, size_t *arr, size_t len){
  Tensor ret       = t;
  ret.n            = len;
  ret.type         = RESHAPE;
  ret.data_offset  = t.data_offset + offset;

  size_t *new_dims = malloc(sizeof(size_t) * len);
  size_t *new_strd = malloc(sizeof(size_t) * len);
  memcpy(new_dims, arr, sizeof(size_t) * len);

  ret.dims         = new_dims;
  ret.strides      = new_strd;

  for(int i = 0; i < len; i++)
    ret.strides[i] = tensor_axis_stride(ret, ret.n - i - 1);
  ret.size = ret.strides[0] * ret.dims[0];

  return ret;
}

/*
 * Performs an elementwise addition on two tensors and stores the result
 * in a third tensor.
 */
void tensor_elementwise_add(const Tensor a, const Tensor b, Tensor c){
  if(a.n != b.n || a.n != c.n)
    SK_ERROR("Tensors must have same number of dims (%lu vs %lu vs %lu).", a.n, b.n, c.n);
    
  for(int i = 0; i < a.n; i++)
    if(a.dims[i] != b.dims[i] || a.dims[i] != c.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu vs %lu\n", i, a.dims[i], b.dims[i], c.dims[i]);

  size_t num_iters = 1;
  for(int i = 0; i < a.n; i++)
    num_iters *= a.dims[i];
  
  size_t pos[a.n];
  for(int i = 0; i < a.n; i++)
    pos[i] = 0;

  float *src_a  = tensor_raw(a);
  float *src_b  = tensor_raw(b);
  float *dest_c = tensor_raw(c);

  for(int i = 0; i < num_iters; i++){
    float one = src_a[tensor_flat_idx(a, pos, a.n)];
    float two = src_b[tensor_flat_idx(b, pos, b.n)];
    dest_c[tensor_flat_idx(c, pos, c.n)] = one + two;
    
    pos[a.n - 1]++;
    for(int i = a.n - 1; i > 0; i--){
      if(!(pos[i] % a.dims[i])){
        pos[i-1]++;
        pos[i] = 0;
      }else break;
    }
  }
}

/*
 * Performs an elementwise addition on two tensors and stores the result
 * in a third tensor.
 */
void tensor_elementwise_sub(const Tensor a, const Tensor b, Tensor c){
  if(a.n != b.n || a.n != c.n)
    SK_ERROR("Tensors must have same number of dims (%lu vs %lu vs %lu).", a.n, b.n, c.n);
    
  for(int i = 0; i < a.n; i++)
    if(a.dims[i] != b.dims[i] || a.dims[i] != c.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu vs %lu\n", i, a.dims[i], b.dims[i], c.dims[i]);

  size_t num_iters = 1;
  for(int i = 0; i < a.n; i++)
    num_iters *= a.dims[i];
  
  size_t pos[a.n];
  for(int i = 0; i < a.n; i++)
    pos[i] = 0;

  float *src_a  = tensor_raw(a);
  float *src_b  = tensor_raw(b);
  float *dest_c = tensor_raw(c);

  for(int i = 0; i < num_iters; i++){
    float one = src_a[tensor_flat_idx(a, pos, a.n)];
    float two = src_b[tensor_flat_idx(b, pos, b.n)];
    dest_c[tensor_flat_idx(c, pos, c.n)] = one - two;
    
    pos[a.n - 1]++;
    for(int i = a.n - 1; i > 0; i--){
      if(!(pos[i] % a.dims[i])){
        pos[i-1]++;
        pos[i] = 0;
      }else break;
    }
  }
}

/*
 * Performs an elementwise multiplication on two tensors and stores the result
 * in a third tensor.
 */
void tensor_elementwise_mul(const Tensor a, const Tensor b, Tensor c){
  if(a.n != b.n || a.n != c.n)
    SK_ERROR("Tensors must have same number of dims (%lu vs %lu vs %lu).", a.n, b.n, c.n);
    
  for(int i = 0; i < a.n; i++)
    if(a.dims[i] != b.dims[i] || a.dims[i] != c.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu vs %lu\n", i, a.dims[i], b.dims[i], c.dims[i]);

  size_t num_iters = 1;
  for(int i = 0; i < a.n; i++)
    num_iters *= a.dims[i];
  
  size_t pos[a.n];
  for(int i = 0; i < a.n; i++)
    pos[i] = 0;

  float *src_a  = tensor_raw(a);
  float *src_b  = tensor_raw(b);
  float *dest_c = tensor_raw(c);

  for(int i = 0; i < num_iters; i++){
    float one = src_a[tensor_flat_idx(a, pos, a.n)];
    float two = src_b[tensor_flat_idx(b, pos, b.n)];
    dest_c[tensor_flat_idx(c, pos, c.n)] = one * two;
    
    pos[a.n - 1]++;
    for(int i = a.n - 1; i > 0; i--){
      if(!(pos[i] % a.dims[i])){
        pos[i-1]++;
        pos[i] = 0;
      }else break;
    }
  }
}

/*
 * Multiplies the contents of a tensor by a scalar value.
 */
void tensor_scalar_mul(Tensor t, float a, Tensor dest){
  if(t.n != dest.n)
    SK_ERROR("Tensors must have same number of dims (%lu vs %lu).", t.n, dest.n);
    
  for(int i = 0; i < t.n; i++)
    if(t.dims[i] != dest.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu\n", i, t.dims[i], dest.dims[i]);

  if(t.device == SIEKNET_CPU){
    size_t pos[t.n];
    memset(pos, '\0', sizeof(size_t)*t.n);

    for(int i = 0; i < t.size; i++){
      tensor_raw(dest)[tensor_flat_idx(dest, pos, dest.n)] = tensor_raw(t)[tensor_flat_idx(t, pos, t.n)] * a;
      pos[t.n - 1]++;
      for(int j = t.n - 1; j > 0; j--){
        if(!(pos[j] % t.dims[j])){
          pos[j-1]++;
          pos[j] = 0;
        }else break;
      }
    }
  }else
    SK_ERROR("Not supported.");
}

/*
 * Performs a matrix multiplication given two 2d tensors, 
 * storing the result in a third 2d tensor.
 */
void tensor_mmult(const Tensor a, const Tensor b, Tensor c){
#ifdef SIEKNET_DEBUG
  if(a.n > 2 || b.n > 2 || c.n > 2)
    SK_ERROR("Dimensions of all matrices must be 2 or fewer - got dimensions %lu, %lu, %lu.\n", a.n, b.n, c.n);
#endif
  
  int left_axis_a = a.n == 2 ? a.n - 2 : 0;
  int right_axis_a = a.n == 2 ? a.n - 1 : -1;

  int left_axis_b = b.n == 2 ? b.n - 2 : 0;
  int right_axis_b = b.n == 2 ? b.n - 1 : -1;

  int left_axis_c = c.n == 2 ? c.n - 2 : 0;
  int right_axis_c = c.n == 2 ? c.n - 1 : -1;

  size_t left_dim_a  =  left_axis_a >= 0 ? a.dims[left_axis_a]  : 1;
  size_t right_dim_a = right_axis_a >= 0 ? a.dims[right_axis_a] : 1;
  
  size_t left_dim_b  =  left_axis_b >= 0 ? b.dims[left_axis_b]  : 1;
  size_t right_dim_b = right_axis_b >= 0 ? b.dims[right_axis_b] : 1;

  size_t left_dim_c  =  left_axis_c >= 0 ? c.dims[left_axis_c]  : 1;
  size_t right_dim_c = right_axis_c >= 0 ? c.dims[right_axis_c] : 1;

  size_t left_stride_a  =  left_axis_a >= 0 ? a.strides[left_axis_a]  : 0;
  size_t right_stride_a = right_axis_a >= 0 ? a.strides[right_axis_a] : 0;
  
  size_t left_stride_b  =  left_axis_b >= 0 ? b.strides[left_axis_b]  : 0;
  size_t right_stride_b = right_axis_b >= 0 ? b.strides[right_axis_b] : 0;

  size_t left_stride_c  =  left_axis_c >= 0 ? c.strides[left_axis_c]  : 0;
  size_t right_stride_c = right_axis_c >= 0 ? c.strides[right_axis_c] : 0;

  if(right_dim_a != left_dim_b && b.n == 1){
    SWAP(left_dim_b, right_dim_b);
    SWAP(left_stride_b, right_stride_b);
  }

#ifdef SIEKNET_DEBUG
  if(right_dim_a != left_dim_b)
    SK_ERROR("Tensor dimensions must match - got (%lu x %lu) * (%lu x %lu).", left_dim_a, right_dim_a, left_dim_b, right_dim_b);

  if(left_dim_a != left_dim_c || right_dim_b != right_dim_c)
    SK_ERROR("Output tensor dimension must match - expected (%lu x %lu) but got (%lu x %lu).", left_dim_a, right_dim_b, left_dim_c, right_dim_c);
#endif

  float *raw_a = tensor_raw(a);
  float *raw_b = tensor_raw(b);
  float *raw_c = tensor_raw(c);

  for(int i = 0; i < left_dim_a; i++){
    for(int j = 0; j < right_dim_b; j++){
      for(int k = 0; k < left_dim_b; k++){
        float a_ik = raw_a[i * left_stride_a  + k * right_stride_a];
        float b_jk = raw_b[j * right_stride_b + k * left_stride_b];
        raw_c[i * left_stride_c + j * right_stride_c] += a_ik * b_jk;
      }
    }
  }
}

void arr_to_tensor(float *buff, size_t bufflen, Tensor t, size_t *arr, size_t len){
  if(len != t.n - 1)
    SK_ERROR("Expected %lu indices for a %lu-dimensional tensor, but got %lu.", t.n - 1, t.n, len);

  if(bufflen != t.dims[t.n - 1])
    SK_ERROR("Buffer must be of length %lu (got %lu) for tensor with rightmost dimension %lu", t.dims[t.n-1], bufflen, t.dims[t.n-1]);

  if(t.device == SIEKNET_CPU){
    float *dest = &((float*)t.data)[tensor_flat_idx(t, arr, len)];
    memcpy(dest, buff, sizeof(float) * bufflen);
  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("GPU currently not supported.");
  }
}

/* 
 * Creates a new tensor given an input array of shapes.
 */
Tensor tensor_from_arr(TENSOR_DEVICE device, size_t *dimensions, size_t num_dimensions){
  size_t num_elements = 1;
  for(int i = 0; i < num_dimensions; i++)
    num_elements *= dimensions[i];

  Tensor ret = {0};
  ret.n = num_dimensions;
  ret.device = device;
  ret.data_offset = 0;
  ret.type = TENSOR;
  if(device == SIEKNET_CPU){
    ret.dims    = calloc(num_dimensions, sizeof(size_t));
    ret.strides = calloc(num_dimensions, sizeof(size_t));
    ret.data    = calloc(num_elements, sizeof(float));
  }else{
    SK_ERROR("GPU currently not supported.");
  }

  memcpy(ret.dims, dimensions, num_dimensions * sizeof(size_t));
  for(int i = 0; i < ret.n; i++){
    ret.strides[i] = tensor_axis_stride(ret, ret.n - i - 1);
  }
  ret.size = ret.strides[0] * ret.dims[0];
  return ret;
}

/*
 * Deallocates a tensor from the heap.
 */
void tensor_dealloc(Tensor t){
  switch(t.type){
    case TENSOR:
      free(t.dims);
      free(t.strides);
      if(t.device == SIEKNET_CPU)
        free(t.data);
      else
        SK_ERROR("tensor_dealloc not implemented for gpu.");
      break;
    case SUBTENSOR:
      break;
    case RESHAPE:
      free(t.dims);
      free(t.strides);
      break;
    default:
      break;
  }
}

/*
 * Pretty-prints a tensor to the terminal.
 */
void tensor_print(Tensor t){
  printf("Tensor: (");
  for(int i = 0; i < t.n; i++){
    printf("%lu", t.dims[i]);
    if(i < t.n - 1) printf(" x ");
    //else printf(") - %lu reals\n", t.size);
    else printf(")\n");
  }

  size_t pos[t.n];
  memset(pos, '\0', t.n * sizeof(size_t));

  printf("{\n");
  while(1){
    for(int i = 1; i < t.n - 1; i++){
      if(!(pos[i] % t.dims[i])){
        int new_row = 1;
        for(int j = i; j < t.n-1; j++)
          if((pos[j] % t.dims[j]))
            new_row = 0;
        if(new_row){
          for(int j = 0; j < i; j++)
            printf("  ");
          printf("{\n");
        }
      }
    }

    for(int i = 0; i < t.n; i++)
      printf("  ");

    printf("{ ");
    for(int i = 0; i < t.dims[t.n - 1]; i++){
      pos[t.n - 1] = i;
      printf("%8.5f", tensor_raw(t)[tensor_flat_idx(t, pos, t.n)]);
      if(i < t.dims[t.n - 1] - 1) printf(", ");
      else printf(" }\n");
    }

    pos[t.n - 2]++;
    for(int i = t.n - 2; i > 0; i--){
      if(!(pos[i] % t.dims[i])){
        pos[i-1]++;
        pos[i] = 0;
        for(int j = 0; j < i; j++)
          printf("  ");
        printf("}\n");
      }else break;
    }

    if(pos[0] == t.dims[0]){
      printf("}\n");
      break;
    }

    if(t.n == 1) break;
  }
}
