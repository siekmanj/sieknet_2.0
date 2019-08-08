#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <tensor.h>
#include <util.h>

size_t tensor_flat_idx(Tensor t, size_t *arr, size_t len){
  size_t idx = t.data_offset;

  size_t max_idx = MIN(len, t.n);
  for(int i = 0; i < max_idx; i++)
    idx += t.strides[i] * arr[i];

  return idx;
}

float tensor_at_idx(Tensor t, size_t *arr, size_t len){
  if(t.device == SIEKNET_CPU){
    return ((float*)t.data)[tensor_flat_idx(t, arr, len)];

  }else if(t.device == SIEKNET_GPU){

  }
  return -1;
}

void tensor_fill_random(Tensor t, float mean, float std){
  float *reals = (float*)t.data;
  size_t num_reals = 1;
  
  for(int i = 0; i < t.n; i++)
    num_reals *= t.dims[i];

  for(int i = 0; i < num_reals; i++)
    reals[i] = normal(mean, std);
}

void tensor_zero(Tensor t){
  if(t.device == SIEKNET_CPU){
    size_t len = 1;
    for(int i = 0; i < t.n; i++)
      len *= t.dims[i];
    float *x = &((float *)t.data)[t.data_offset];
    for(int i = 0; i < len; i++)
      x[i] = 0.0f;

  }else{
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }
}

Tensor tensor_clone(Tensor src){
  Tensor ret;

  return ret;
}

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

  if(src.device == SIEKNET_CPU){
    float *src_mem  = &((float *)src.data)[src.data_offset];
    float *dest_mem = &((float *)dest.data)[dest.data_offset];

    for(int i = 0; i < len; i++)
      dest_mem[i] = src_mem[i];
    
  }else{
    SK_ERROR("Not implemented.");
  }
}

void tensor_sigmoid_precompute(Tensor t, Tensor d){
  if(t.n > 1)
    SK_ERROR("Logistics not supported for non-1d tensors.");

  if(d.data != NULL && t.n != d.n)
    SK_ERROR("If derivative tensor is supplied, dimensions must match. T dims: %lu, d dims: %lu", t.n, d.n);

  for(int i = 0; i < t.n && d.data != NULL; i++)
    if(t.dims[i] != d.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu\n", i, t.dims[i], d.dims[i]);

  if(t.device == SIEKNET_CPU){
    float *z_mem = &((float *)t.data)[t.data_offset];
    float *d_mem = d.data != NULL ? &((float *)d.data)[d.data_offset] : NULL;
    size_t z_str = t.strides[0];
    size_t d_str = d.strides[0];

    for(int i = 0; i < t.dims[0]; i++){
      z_mem[i * z_str] = 1 / (1 + exp(-z_mem[i * z_str]));
      if(d_mem)
        d_mem[i * d_str] = z_mem[i * z_str] * (1 - z_mem[i * z_str]);
    }

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_tanh_precompute(Tensor t, Tensor d){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_relu_precompute(Tensor t, Tensor d){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_selu_precompute(Tensor t, Tensor d){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}


void tensor_softmax_precompute(Tensor t, Tensor d){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Not implemented.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

float tensor_quadratic_cost(Tensor o, Tensor y, Tensor grad){
  if(y.device != o.device || y.device != grad.device)
    SK_ERROR("Devices must match.");

  if(y.n != 1 || o.n != 1 || grad.n != 1)
    SK_ERROR("All tensor dimensions must be 1. Got %lu, %lu, %lu.", y.n, o.n, grad.n);

  if(y.dims[0] != o.dims[0] || y.dims[0] != grad.dims[0])
    SK_ERROR("All tensors must be of the same length, got lengths %lu, %lu, %lu", y.dims[0], o.dims[0], grad.dims[0]);

  float cost = 0;
  if(y.device == SIEKNET_CPU){

    float *o_mem = &((float *)o.data)[o.data_offset];
    float *y_mem = &((float *)y.data)[y.data_offset];
    float *g_mem = &((float *)grad.data)[grad.data_offset];
    for(int i = 0; i < y.dims[0]; i++){
      float o_i = o_mem[i * o.strides[0]];
      float y_i = y_mem[i * y.strides[0]];

      cost += 0.5 * (o_i - y_i) * (o_i - y_i);
      g_mem[i * grad.strides[0]] = (o_i - y_i);
      printf("grad: %f - %f\n", o_i, y_i);
    }
    return cost;
  }else if(y.device == SIEKNET_GPU){
    SK_ERROR("Tensor cost not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

float tensor_cross_entropy_cost(Tensor y, Tensor label, Tensor grad){
  if(y.device != label.device || y.device != grad.device)
    SK_ERROR("Devices must match.");

  if(y.n != 1 || label.n != 1 || grad.n != 1)
    SK_ERROR("All tensor dimensions must be 1. Got %lu, %lu, %lu.", y.n, label.n, grad.n);

  if(y.dims[0] != label.dims[0] || y.dims[0] != grad.dims[0])
    SK_ERROR("All tensors must be of the same length, got lengths %lu, %lu, %lu", y.dims[0], label.dims[0], grad.dims[0]);

  float cost = 0;
  if(y.device == SIEKNET_CPU){

    /*
    float *y_mem = &((float *)y.data)[y.data_offset];
    float *l_mem = &((float *)label.data)[label.data_offset];
    float *g_mem = &((float *)grad.data)[grad.data_offset];
    for(int i = 0; i < y.dims[0]; i++){
      float y_i = y_mem[i * y.strides[0]];
      float l_i = l_mem[i * label.strides[0]];

      cost += (y_i - l_i) * (y_i - l_i);
      g_mem[i * grad.strides[0]] = y_i - l_i;
    }
    */
    return cost;
  }else if(y.device == SIEKNET_GPU){
    SK_ERROR("Tensor cost not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_transpose(Tensor t, size_t dim1, size_t dim2){
  if(dim1 > t.n || dim2 > t.n)
    SK_ERROR("Invalid axes (%lu, %lu) for tensor with dimension %lu.", dim1, dim2, t.n);

  SWAP(t.dims[t.n - dim1 - 1], t.dims[t.n - dim2 - 1]);
  SWAP(t.strides[t.n - dim1 - 1], t.strides[t.n - dim2 - 1]);

}

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
    ret.data_offset  = tensor_flat_idx(t, arr, len);
  }else{
    ret.n            = t.n - len;
    ret.dims         = &t.dims[len];
    ret.strides      = &t.strides[len];
    ret.data_offset  = tensor_flat_idx(t, arr, len);
  }
  return ret;
}

static size_t tensor_axis_stride(Tensor t, size_t axis){
  size_t stride = 1;
  for(int i = 0; i < axis; i++)
    stride *= t.dims[t.n - i - 1];
  return stride;
}

/*
 * TODO: Tensor broadcasting?
 * TODO: GPU support
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

  float *src_a  = (float*)a.data;
  float *src_b  = (float*)b.data;
  float *dest_c = (float*)c.data;

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

  float *src_a  = (float*)a.data;
  float *src_b  = (float*)b.data;
  float *dest_c = (float*)c.data;

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


void tensor_mmult(const Tensor a, const Tensor b, Tensor c){
#ifdef SIEKNET_DEBUG
  if(a.n > 2 || b.n > 2 || c.n > 2)
    SK_ERROR("Dimensions of all matrices must be 2 or fewer - got dimensions %lu, %lu, %lu.\n", a.n, b.n, c.n);
#endif
  
  size_t outer_dim_a = a.n == 2 ? a.dims[a.n - 2] : a.dims[0];
  size_t inner_dim_a = a.n == 2 ? a.dims[a.n - 1] : 1;

  size_t outer_dim_b = b.n == 2 ? b.dims[b.n - 2] : b.dims[0];
  size_t inner_dim_b = b.n == 2 ? b.dims[b.n - 1] : 1;

  size_t outer_dim_c = c.n == 2 ? c.dims[c.n - 2] : c.dims[0];
  size_t inner_dim_c = c.n == 2 ? c.dims[c.n - 1] : 1;

#ifdef SIEKNET_DEBUG
  if(inner_dim_a != outer_dim_b)
    SK_ERROR("Tensor dimensions must match - got (%lu x %lu) * (%lu x %lu).", outer_dim_a, inner_dim_a, outer_dim_b, inner_dim_b);

  if(outer_dim_a != outer_dim_c || inner_dim_b != inner_dim_c)
    SK_ERROR("Output tensor dimension must match - expected (%lu x %lu) but got (%lu x %lu).", outer_dim_a, inner_dim_b, outer_dim_c, inner_dim_c);
#endif

  if(b.n > 1)
    tensor_transpose(b, 0, 1); // swap axes 0 and 1

  for(int i = 0; i < outer_dim_a; i++){
    Tensor w, x, y;
    if(a.n > 1)
      w = get_subtensor(a, i);
    else
      w = a;

    for(int j = 0; j < inner_dim_b; j++){
      if(b.n > 1)
        x = get_subtensor(b, j);
      else
        x = b;

      if(c.n > 1)
        y = get_subtensor(c, i, j);
      else
        y = get_subtensor(c, i);

      ((float*)y.data)[y.data_offset] += tensor_reduce_dot(w, x);

    }
  }
  if(b.n > 1)
    tensor_transpose(b, 0, 1); // un-swap axes 0 and 1
}

/*
 * Perform a dot-product reduction on two tensors,
 * CPU only.
 */
float tensor_reduce_dot(const Tensor a, const Tensor b){
#ifdef SIEKNET_DEBUG
  if(a.n != 1 || b.n != 1)
    SK_ERROR("Dot product not defined for tensors greater than dimension 1 (got dimensions %lu and %lu).\n", a.n, b.n);

  if(a.dims[0] != b.dims[0])
    SK_ERROR("Dot product not defined for tensors of unequal length (%lu vs %lu)\n", a.dims[0], b.dims[0]);
#endif

  float ret = 0;
  if(a.device == SIEKNET_CPU){
    const float *w = &((float*)a.data)[a.data_offset];
    const float *x = &((float*)b.data)[b.data_offset];

    for(int i = 0; i < b.dims[0]; i++){
      ret += w[a.strides[0]* i] * x[b.strides[0]* i];
    }
    return ret;
  }

  if(a.device == SIEKNET_GPU)
    SK_ERROR("tensor_reduce_dot can only be run on the CPU.");
  else
    SK_ERROR("Invalid device.");

  return 0;
}

void arr_to_tensor(float *buff, size_t bufflen, Tensor t, size_t *arr, size_t len){
  if(len != t.n - 1)
    SK_ERROR("Expected %lu indices for a %lu-dimensional tensor, but got %lu.", t.n - 1, t.n, len);

  if(bufflen != t.dims[t.n - 1])
    SK_ERROR("Buffer must be of length %lu (got %lu) for tensor with innermost dimension %lu", t.dims[t.n-1], bufflen, t.dims[t.n-1]);

  if(t.device == SIEKNET_CPU){
    float *dest = &((float*)t.data)[tensor_flat_idx(t, arr, len)];
    memcpy(dest, buff, sizeof(float) * bufflen);
  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("GPU currently not supported.");
  }
}

Tensor tensor_from_arr(Device device, size_t *dimensions, size_t num_dimensions){
  size_t num_elements = 1;
  for(int i = 0; i < num_dimensions; i++)
    num_elements *= dimensions[i];

  Tensor ret = {0};
  ret.n = num_dimensions;
  ret.device = device;
  ret.data_offset = 0;
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
  return ret;
}

void tensor_print(Tensor t){
  printf("Tensor: (");
  for(int i = 0; i < t.n; i++){
    printf("%lu", t.dims[i]);
    if(i < t.n - 1) printf(" x ");
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
      printf("%6.4f", ((float *)t.data)[tensor_flat_idx(t, pos, t.n)]);
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
