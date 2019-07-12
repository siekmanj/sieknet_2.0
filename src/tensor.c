#include <tensor.h>

Tensor arr_to_tensor(void *arr, size_t *dims, size_t ndim){
  Tensor t;
  switch(ndim){
    case 0:{
      SK_ERROR("Cannot have 0 dims.");
      break;
    }
    case 1:{
      float *d = (float *)arr;

      break;
    }
    case 2:{
      float **d = (float **)arr;

      break;
    }
    case 3:{
      float ***d = (float ***)arr;

      break;
    }
    default:{
      SK_ERROR("Dimensions more than three not supported.");
    }
  }
  return t;
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
