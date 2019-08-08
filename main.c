#include <stdio.h>

#include <sieknet.h>
#include <tensor.h>
#include <util.h>

const char* test0= "model/one.sk";
const char* test1= "model/two.sk";
const char* test2= "model/three.sk";

int main(){
  srand(0);
  {
    printf("TRANSPOSE_TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 15, 16);
    Tensor b = create_tensor(SIEKNET_CPU, 15, 16);
    tensor_fill_random(a);
    tensor_copy(a, b);
    tensor_transpose(b, 1, 0);
    float *a_raw = &((float*)a.data)[a.data_offset];
    float *b_raw = &((float*)b.data)[b.data_offset];
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
  }

  {
    printf("ELEMENTWISE_ADD: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a);
    tensor_fill_random(b);
    float *a_raw = &((float*)a.data)[a.data_offset];
    float *b_raw = &((float*)b.data)[b.data_offset];
    float *c_raw = &((float*)c.data)[c.data_offset];
    
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
  }

  {
    printf("ELEMENTWISE_MUL: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a);
    tensor_fill_random(b);
    float *a_raw = &((float*)a.data)[a.data_offset];
    float *b_raw = &((float*)b.data)[b.data_offset];
    float *c_raw = &((float*)c.data)[c.data_offset];
    
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
  }

  {
    printf("GRADIENT CHECK: ");
    Network n = sk_create_network(test0);
    Tensor x = create_tensor(SIEKNET_CPU, 25, n.input_dimension);
    Tensor y = create_tensor(SIEKNET_CPU, 25, n.layers[n.depth-1]->output.dims[1]);

    float *params = &((float *)n.params.data)[n.params.data_offset];
    float *p_grad = &((float *)n.param_grad.data)[n.param_grad.data_offset];
    for(int i = 0; i < n.num_params; i++){
      tensor_fill_random(x);
      tensor_fill_random(y);
      sk_forward(&n, x);
      sk_cost(&n, n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      sk_backward(&n);
      printf("param %d: %f, grad %d: %f\n", i, params[i], i, p_grad[i]);
    }
  }

#if 0
  Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
  tensor_fill_random(a);
  tensor_print(a);

  tensor_transpose(a, 0, 2);
  tensor_print(a);
#endif
#if 0

  Tensor a = create_tensor(SIEKNET_CPU, 2, 3);
  tensor_fill_random(a);
  tensor_print(a);
  
  Tensor b = create_tensor(SIEKNET_CPU, 3, 4);
  tensor_fill_random(b);
  tensor_print(b);

  Tensor c = create_tensor(SIEKNET_CPU, 2, 4);
  tensor_print(c);

  tensor_mmult(a, b, c);

  tensor_print(c);

#endif
  Network n = sk_create_network(test0);
#if 0

  Tensor x1 = create_tensor(SIEKNET_CPU, n.input_dimension);
  Tensor x2 = create_tensor(SIEKNET_CPU, n.input_dimension);
  Tensor x3 = create_tensor(SIEKNET_CPU, n.input_dimension);

  Tensor y1 = create_tensor(SIEKNET_CPU, n.layers[1]->size);
  Tensor y2 = create_tensor(SIEKNET_CPU, n.layers[1]->size);
  Tensor y3 = create_tensor(SIEKNET_CPU, n.layers[1]->size);

  tensor_fill_random(x1);
  tensor_fill_random(x2);
  tensor_fill_random(x3);

  tensor_fill_random(y1);
  tensor_fill_random(y2);
  tensor_fill_random(y3);
#endif

#if 0
  sk_forward(&n, x1);
  float c1 = sk_cost(&n, n.layers[1], y1, SK_QUADRATIC_COST);
  sk_forward(&n, x2);
  float c2 = sk_cost(&n, n.layers[1], y2, SK_QUADRATIC_COST);
  sk_forward(&n, x3);
  float c3 = sk_cost(&n, n.layers[1], y3, SK_QUADRATIC_COST);

  printf("cost 1: %f\n", c1 + c2 + c3);
#endif
#if 1
  Tensor x = create_tensor(SIEKNET_CPU, 13, n.input_dimension);
  Tensor y = create_tensor(SIEKNET_CPU, 13, n.layers[1]->size);
  tensor_fill_random(x);
  tensor_fill_random(y);
  sk_forward(&n, x);
  printf("cost 2: %f\n", sk_cost(&n, n.layers[1], y, SK_QUADRATIC_COST));
  sk_backward(&n);

#endif
  /*
  sk_forward(&n, x1);
  sk_forward(&n, x2);
  sk_forward(&n, x3);
  printf("%lu\n", n.t);
  sk_forward(&n, x);
  printf("two:\n");
  Network two = create_network(test1);
  printf("three:\n");
  Network thr = create_network(test2);
  */
  return 0;
}
