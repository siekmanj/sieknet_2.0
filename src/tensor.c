#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tensor.h>

void tensor_zero(Tensor *t, size_t idx, size_t len){
  if(t->device == SIEKNET_CPU){
    for(int i = idx; i < len; i++)
      t->data[i] = 0.0f;

  }else{
    SK_ERROR("Tensor zeroing not implemented on GPU.");
  }

}

void tensor_inner_product(Tensor *a, size_t idx_a, Tensor *b, size_t idx_b, Tensor *c, size_t idx_c, size_t len){
  if(a->device != b->device || a->device != c->device)
    SK_ERROR("Devices don't match between tensors a, b, and c.");

  if(a->device == SIEKNET_CPU){
    for(int i = 0; i < len; i++)
      c->data[idx_c + i] = a->data[idx_a + i] * b->data[idx_b + i];

  }else{
    SK_ERROR("Dot product not implemented on GPU.");
  }


}

void arr_to_tensor(Tensor t, float *buff, size_t *arr, size_t len){
  if(t.device == SIEKNET_CPU){
    if(len != t.n - 1)
      SK_ERROR("Expected %lu indices for a %lu-dimensional tensor.", t.n - 1, t.n);

    size_t idx = get_flat_idx(t, arr, len);
    printf("idx: %lu\n", idx);

  }else if(t.device == SIEKNET_GPU){
    SK_ERROR("GPU currently not supported.");
  }
}

static size_t get_flat_idx(Tensor t, size_t *arr, size_t len){
  size_t idx = 0;
  size_t memsizes[len];

  for(int i = t.n-1; i >= 0; i--){
    if(i == t.n-1)
      memsizes[i] = t.dims[i];
    else
      memsizes[i] = memsizes[i+1] * t.dims[i+1];
  }

  for(int i = 0; i < t.n-1; i++)
    idx += memsizes[i+1] * arr[i];

  return idx;
}


Tensor tensor_from_arr(Device device, size_t *dimensions, size_t num_dimensions){
  size_t num_elements = 1;
  for(int i = 0; i < num_dimensions; i++)
    num_elements *= dimensions[i];

  Tensor ret = {0};
  ret.n = num_dimensions;
  ret.device = device;
  if(device == SIEKNET_CPU){
    ret.dims = calloc(num_dimensions, sizeof(size_t));
    ret.data = calloc(num_elements, sizeof(float));
  }else{
    SK_ERROR("GPU currently not supported.");
  }

  memcpy(ret.dims, dimensions, num_dimensions * sizeof(size_t));
  return ret;
}
