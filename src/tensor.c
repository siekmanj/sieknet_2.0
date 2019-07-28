#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tensor.h>
#include <util.h>

size_t get_flat_idx(Tensor t, size_t *arr, size_t len){
  size_t idx = 0;

  for(int i = 0; i < len; i++)
    idx += t.strides[i] * arr[i];

  return idx;
}

void tensor_fill_random(Tensor t){
  float *reals = (float*)t.data;
  size_t num_reals = 1;
  
  for(int i = 0; i < t.n; i++)
    num_reals *= t.dims[i];

  for(int i = 0; i < num_reals; i++)
    reals[i] = normal(0, 1);
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

void tensor_sigmoid(Tensor t){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_tanh(Tensor t){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_relu(Tensor t){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_selu(Tensor t){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }else{
    SK_ERROR("Invalid device.");
  }
}

void tensor_softmax(Tensor t){
  if(t.device == SIEKNET_CPU){

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("Tensor zeroing not implemented on GPU.");
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
    ret.data_offset += get_flat_idx(t, arr, len);
  }else{
    ret.n            = t.n - len;
    ret.dims         = &t.dims[len];
    ret.strides      = &t.strides[len];
    ret.data_offset += get_flat_idx(t, arr, len);
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
 */
void tensor_elementwise_add(const Tensor a, const Tensor b, Tensor c){
  if(a.n != b.n || a.n != c.n)
    SK_ERROR("Tensors must have same number of dims (%lu vs %lu vs %lu).", a.n, b.n, c.n);
    
  for(int i = 0; i < a.n; i++)
    if(a.dims[i] != b.dims[i] || a.dims[i] != c.dims[i])
      SK_ERROR("Tensor dimensions do not match on dimension %d: %lu vs %lu vs %lu\n", i, a.dims[i], b.dims[i], c.dims[i]);
  
  if(a.n > 1)
    SK_ERROR("Tensor addition above 1 dimension not supported.");

  float *src_a  = &((float*)a.data)[a.data_offset];
  float *src_b  = &((float*)b.data)[b.data_offset];
  float *dest_c = &((float*)c.data)[c.data_offset];
  for(int i = 0; i < a.dims[0]; i++){
    dest_c[i * c.strides[0]] = src_a[i * a.strides[0]] + src_b[i * b.strides[0]];
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

  if(a.device == SIEKNET_GPU){
    SK_ERROR("tensor_reduce_dot can only be run on the CPU.");
  }

  else{
    SK_ERROR("Invalid device.");
  }
  return 0;
}

void arr_to_tensor(float *buff, size_t bufflen, Tensor t, size_t *arr, size_t len){
  if(len != t.n - 1)
    SK_ERROR("Expected %lu indices for a %lu-dimensional tensor, but got %lu.", t.n - 1, t.n, len);

  if(bufflen != t.dims[t.n - 1])
    SK_ERROR("Buffer must be of length %lu (got %lu) for tensor with innermost dimension %lu", t.dims[t.n-1], bufflen, t.dims[t.n-1]);

  if(t.device == SIEKNET_CPU){
    float *dest = &((float*)t.data)[t.data_offset + get_flat_idx(t, arr, len)];
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
      printf("%6.4f", ((float *)t.data)[t.data_offset + get_flat_idx(t, pos, t.n)]);
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
