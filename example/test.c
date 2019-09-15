#include <stdio.h>
#include <locale.h>

#include <sieknet.h>
#include <tensor.h>

const char* test= "model/test.sk";

int main(){
  srand(1271998);
  setlocale(LC_NUMERIC, "");

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("Unit tests for tensor math library.\n");


#if 0
  float a_mem[] = {-1.0, -1.0, 1.0};
	float c_mem[] = {0.0, -9.3890561, 0.0};
  Tensor a = create_tensor(SIEKNET_CPU, 3);
  Tensor b = create_tensor(SIEKNET_CPU, 3, 3);
	Tensor c = create_tensor(SIEKNET_CPU, 3);
	Tensor d = create_tensor(SIEKNET_CPU, 3);
	Tensor e = create_tensor(SIEKNET_CPU, 3);
  a.data = a_mem;
	c.data = c_mem;
  tensor_softmax_precompute(a, b);
  tensor_print(a);
  tensor_print(b);
	tensor_mmult(b, c, d);
	tensor_print(d);
  exit(1);
#endif
  {
    printf("%-50s", "TENSOR_SCALAR_MUL TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 4, 5);
    tensor_fill_random(a, 0, 1);
    tensor_copy(a, b);
    tensor_scalar_mul(b, 0.1, b);

    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        for(int k = 0; k < 5; k++){
          if(tensor_at(a, i, j, k) == 0.1 * tensor_at(b, i, j, k))
            success = 0;
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }
  {
    printf("%-50s", "SOFTMAX_TEST");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 4, 5, 5);
    tensor_fill_random(a, 0, 1);
    tensor_softmax_precompute(a, b);
    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        float sum = 0;
        for(int k = 0; k < 5; k++){
          sum += tensor_at(a, i, j, k);
        }
        float diff = sum - 1.0f;
        if(MAX(diff , -diff) > 1e-5)
          success = 0;
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
  }

  {
    printf("%-50s", "SOFTMAX_2_TEST");
    Tensor a = create_tensor(SIEKNET_CPU, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 5, 5);
    tensor_fill_random(a, 0, 1);
    tensor_softmax_precompute(a, b);
    int success = 1;
    float sum = 0;
    for(int k = 0; k < 5; k++){
      sum += tensor_at(a, k);
    }
    float diff = sum - 1.0f;
    if(MAX(diff , -diff) > 1e-5)
      success = 0;

    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
  }

  {
    printf("%-50s", "TENSOR_COPY_TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 5, 4);
    tensor_fill_random(a, 0, 1);
    tensor_transpose(b, 0, 1);
    tensor_copy(a, b);

    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        for(int k = 0; k < 5; k++){
          if(tensor_at(a, i, j, k) != tensor_at(b, i, j, k))
            success = 0;
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }

  {
    printf("%-50s", "TRANSPOSE_TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 15, 16);
    Tensor b = create_tensor(SIEKNET_CPU, 15, 16);
    tensor_fill_random(a, 0, 1);
    tensor_copy(a, b);
    tensor_transpose(b, 1, 0);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    int success = 1;
    for(int i = 0; i < 15; i++){
      for(int j = 0; j < 16; j++){
        float a_ij = a_raw[tensor_get_offset(a, i, j)];
        float b_ji = b_raw[tensor_get_offset(b, j, i)];
        if(a_ij != b_ji)
          success = 0;
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }

  {
    printf("%-50s", "ELEMENTWISE_ADD: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a, 0, 1);
    tensor_fill_random(b, 0, 1);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    float *c_raw = tensor_raw(c);
    
    tensor_elementwise_add(a, b, c);

    int success = 1;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 5; j++){
        for(int k = 0; k < 6; k++){
          float a_ijk = a_raw[tensor_get_offset(a, i, j, k)];
          float b_ijk = b_raw[tensor_get_offset(b, i, j, k)];
          float c_ijk = c_raw[tensor_get_offset(c, i, j, k)];
          if(a_ijk + b_ijk != c_ijk){
            printf("%f + %f should be %f, got %f\n", a_ijk, b_ijk, a_ijk + b_ijk, c_ijk);
            success = 0;
          }
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
    tensor_dealloc(c);
  }

  {
    printf("%-50s", "ELEMENTWISE_MUL: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a, 0, 1);
    tensor_fill_random(b, 0, 1);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    float *c_raw = tensor_raw(c);
    
    tensor_elementwise_mul(a, b, c);

    int success = 1;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 5; j++){
        for(int k = 0; k < 6; k++){
          float a_ijk = a_raw[tensor_get_offset(a, i, j, k)];
          float b_ijk = b_raw[tensor_get_offset(b, i, j, k)];
          float c_ijk = c_raw[tensor_get_offset(c, i, j, k)];
          if(a_ijk * b_ijk != c_ijk){
            printf("%f + %f should be %f, got %f\n", a_ijk, b_ijk, a_ijk + b_ijk, c_ijk);
            success = 0;
          }
        }
      }
      
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
    tensor_dealloc(c);
  }

  {
    printf("%-50s", "ELEMENTWISE_SUB: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a, 0, 1);
    tensor_fill_random(b, 0, 1);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    float *c_raw = tensor_raw(c);
    
    tensor_elementwise_sub(a, b, c);

    int success = 1;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 5; j++){
        for(int k = 0; k < 6; k++){
          float a_ijk = a_raw[tensor_get_offset(a, i, j, k)];
          float b_ijk = b_raw[tensor_get_offset(b, i, j, k)];
          float c_ijk = c_raw[tensor_get_offset(c, i, j, k)];
          if(a_ijk - b_ijk != c_ijk){
            printf("%f + %f should be %f, got %f\n", a_ijk, b_ijk, a_ijk + b_ijk, c_ijk);
            success = 0;
          }
        }
      }
      
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
    tensor_dealloc(c);
  }

  return 0;
}
