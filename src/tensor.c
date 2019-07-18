#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tensor.h>


void print_tensor(Tensor t){
  printf("Tensor: (");
  for(int i = 0; i < t.n; i++){
    printf("%lu", t.dims[i]);
    if(i < t.n - 1) printf(", ");
    else printf(")\n");
  }

  size_t pos[t.n];
  memset(pos, '\0', t.n * sizeof(size_t));

  while(1){

    for(int i = 0; i < t.n - 1; i++){
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
      printf("%3.2f", ((float *)t.data)[get_flat_idx(t, pos, t.n) + i]);
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

    if(pos[0] == t.dims[0])
      break;
  }
  printf("}\n");
}

size_t get_flat_idx(Tensor t, size_t *arr, size_t len){
  size_t idx = 0;
  size_t memsizes[len];

  for(int i = t.n - 1; i >= 0; i--){
    if(i == t.n - 1)
      memsizes[i] = t.dims[i];
    else
      memsizes[i] = memsizes[i+1] * t.dims[i+1];
  }

  for(int i = 0; i < t.n-1; i++)
    idx += memsizes[i+1] * arr[i];

  return idx;
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

/*
 * Perform a dot-product reduction on two tensors, 
 * and store the result in a third vector.
 */
void tensor_reduce_dot(const Tensor a, size_t idx_a, size_t axis_a,
                       const Tensor b, size_t idx_b, size_t axis_b,
                       Tensor c, size_t idx_c, size_t len){
  print_tensor(a);
  print_tensor(b);
  print_tensor(c);

  if(a.device != b.device || a.device != c.device)
    SK_ERROR("Devices don't match between tensors a, b, and c.");

  if(a.dims[a.n - axis_a - 1] != b.dims[b.n - axis_b - 1])
    SK_ERROR("Tensors have invalid dimensions at axes %lu and %lu - cannot dot product two vectors of sizes %lu and %lu.\n", axis_a, axis_b, a.dims[a.n - axis_a - 1], b.dims[b.n - axis_b - 1]);

  size_t stride_a = 1;
  for(int i = 0; i < axis_a; i++)
    stride_a *= a.dims[a.n - i - 1];

  size_t stride_b = 1;
  for(int i = 0; i < axis_b; i++)
    stride_b *= a.dims[b.n - i - 1];

  if(a.device == SIEKNET_CPU){
    const float *w = &((float*)a.data)[a.data_offset + idx_a];
    const float *x = &((float*)&b.data)[b.data_offset + idx_b];
    float *y       = &((float*)&c.data)[c.data_offset + idx_c];

    for(int i = 0; i < len; i++){
      size_t offset_a = idx_a + i * stride_a;
      size_t offset_b = idx_b + i * stride_b;
      *y += w[offset_a] * x[offset_b];
    }
  }else{
    SK_ERROR("Tensor dot product not implemented on GPU.");
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
