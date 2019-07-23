#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tensor.h>
#include <util.h>


void tensor_print(Tensor t){
  printf("Tensor: (");
  for(int i = 0; i < t.n; i++){
    printf("%lu", t.dims[i]);
    if(i < t.n - 1) printf(", ");
    else printf(")\n");
  }

  size_t pos[t.n];
  memset(pos, '\0', t.n * sizeof(size_t));

  printf("{\n");
  while(1){
#if 0
    printf("pos: (");
    for(int i = 0; i < t.n; i++){
      printf("%lu vs %lu", pos[i], t.dims[i]);
      if(i < t.n - 1) printf(", ");
      else printf(")\n");
    }
#endif

    for(int i = 1; i < t.n - 1; i++){
      if(!(pos[i] % t.dims[i])){
        int new_row = 1;
        for(int j = i; j < t.n-1; j++){
          if((pos[j] % t.dims[j]))
            new_row = 0;
        }

        if(new_row){
          for(int j = 0; j < i; j++){
            printf("  ");
          }
          printf("{\n");
        }
      }
    }

    for(int i = 0; i < t.n; i++){
      printf("  ");
    }

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
        for(int j = 0; j < i; j++){
          printf("  ");
        }
        printf("}\n");
      }else
        break;
    }

    if(pos[0] == t.dims[0]){
      printf("}\n");
      break;
    }

    if(t.n == 1)
      break;
  }
}

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

void tensor_zero(Tensor t, size_t idx, size_t len){
  if(t.device == SIEKNET_CPU){
    float *x = &((float *)t.data)[t.data_offset];
    for(int i = idx; i < len; i++)
      x[i] = 0.0f;

  }else{
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }
}

void tensor_transpose(Tensor t, size_t dim1, size_t dim2){
  if(dim1 > t.n || dim2 > t.n)
    SK_ERROR("Invalid axis (%lu, %lu) for tensor with dimension %lu.", dim1, dim2, t.n);

  SWAP(t.dims[t.n - dim1 - 1], t.dims[t.n - dim2 - 1]);
  SWAP(t.strides[t.n - dim1 - 1], t.strides[t.n - dim2 - 1]);

}


size_t tmp_dim = 1;
size_t tmp_str = 0;
Tensor tensor_to_subtensor(Tensor t, size_t *arr, size_t len){
  Tensor ret = t;
  if(len > t.n)
    SK_ERROR("Cannot get subtensor with dimension less than 1 from tensor of dimension %lu\n", t.n);
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

void tensor_mmult(const Tensor a, const Tensor b, Tensor c){

  if(a.n > 2 || b.n > 2 || c.n > 2)
    SK_ERROR("Dimensions of all matrices must be 2 or fewer - got dimensions %lu, %lu, %lu.\n", a.n, b.n, c.n);
  
  size_t outer_dim_a = a.n == 2 ? a.dims[a.n - 2] : a.dims[0];
  size_t inner_dim_a = a.n == 2 ? a.dims[a.n - 1] : 1;

  size_t outer_dim_b = b.n == 2 ? b.dims[b.n - 2] : b.dims[0];
  size_t inner_dim_b = b.n == 2 ? b.dims[b.n - 1] : 1;

  size_t outer_dim_c = c.n == 2 ? c.dims[c.n - 2] : c.dims[0];
  size_t inner_dim_c = c.n == 2 ? c.dims[c.n - 1] : 1;

  size_t outer_stride_a = a.n == 2 ? a.strides[a.n - 2] : 1;
  size_t inner_stride_a = a.n == 2 ? a.strides[a.n - 1] : 1;

  size_t outer_stride_b = b.n == 2 ? b.strides[b.n - 2] : 1;
  size_t inner_stride_b = b.n == 2 ? b.strides[b.n - 1] : 1;

  size_t outer_stride_c = c.n == 2 ? c.strides[c.n - 2] : 1;
  size_t inner_stride_c = c.n == 2 ? c.strides[c.n - 1] : 1;

  if(inner_dim_a != outer_dim_b)
    SK_ERROR("Tensor dimensions must match - got (%lu x %lu) * (%lu x %lu).", outer_dim_a, inner_dim_a, outer_dim_b, inner_dim_b);

  if(outer_dim_a != outer_dim_c || inner_dim_b != inner_dim_c)
    SK_ERROR("Output tensor dimension must match - expected (%lu x %lu) but got (%lu x %lu).", outer_dim_a, inner_dim_b, outer_dim_c, inner_dim_c);

  tensor_transpose(b, 0, 1);
  for(int i = 0; i < outer_dim_a; i++){
    //size_t offset_a = i * outer_stride_a;
    //printf("GETTING SUBTENSOR FOR I: %lu\n", i);
    Tensor w = get_subtensor(a, i);
    //tensor_print(w);

    for(int j = 0; j < inner_dim_b; j++){
      Tensor x = get_subtensor(b, j);
      Tensor y = get_subtensor(c, i, j);
      ((float*)y.data)[y.data_offset] += tensor_reduce_dot(w, x);
      tensor_print(y);
      //tensor_print(x);
      //printf("Doing c[%lu, %lu]:\n", i, j);

      //size_t offset_b = j * inner_stride_b;
      //size_t offset_c = i * outer_stride_c + j * inner_stride_c;

      //tensor_reduce_dot

      /*
      tensor_reduce_dot(a, offset_a, inner_stride_a,
                        b, offset_b, outer_stride_b,
                        c, offset_c, outer_dim_b);
      */
    }
  }
  tensor_transpose(b, 0, 1); // return x to its original position
}

/*
 * Perform a dot-product reduction on two tensors,
 */
float tensor_reduce_dot(const Tensor a, const Tensor b){

  /*
  float ret = 0;
  if(a.device == SIEKNET_CPU){
    const float *w = &((float*)a.data)[a.data_offset + offset_a];
    const float *x = &((float*)b.data)[b.data_offset + offset_b];

    for(int i = 0; i < len; i++){
      printf("y[%lu] += %f * %f\n", offset_c, w[stride_a * i], x[stride_b * i]);
      ret += w[stride_a * i] * x[stride_b * i];
    }
    return;
  }

  if(a.device == SIEKNET_GPU){
    SK_ERROR("tensor_reduce_dot not yet implemented on GPU.");
    return;

  }

  else{
    SK_ERROR("Invalid device.");
  }
  */
}

void arr_to_tensor(Tensor t, float *buff, size_t *arr, size_t len){
  if(t.device == SIEKNET_CPU){
    if(len != t.n - 1)
      SK_ERROR("Expected %lu indices for a %lu-dimensional tensor, but got %lu.", t.n - 1, t.n, len);

    size_t idx = get_flat_idx(t, arr, len);

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
    printf("dim %lu has stride %lu\n", ret.dims[i], ret.strides[i]);
  }

  return ret;
}
