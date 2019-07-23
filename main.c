#include <stdio.h>

#include <sieknet.h>
#include <tensor.h>
#include <util.h>

const char* test0= "model/one.sk";
const char* test1= "model/two.sk";
const char* test2= "model/three.sk";

int main(){
  srand(0);
#if 0
  Tensor a = create_tensor(SIEKNET_CPU, 4, 3);
  tensor_fill_random(a);
  Tensor a1 = get_subtensor(a, 0);
  Tensor a2 = get_subtensor(a, 1);
  Tensor a3 = get_subtensor(a, 2);

  tensor_print(a);
  tensor_print(a1);
  tensor_print(a2);
  tensor_print(a3);
#endif
#if 0
  Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
  tensor_fill_random(a);
  tensor_print(a);

  tensor_transpose(a, 0, 2);
  tensor_print(a);
#endif
#if 1

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
#if 0
  Network n = sk_create_network(test0);

  float *x = calloc(n.input_dimension, sizeof(float));
  sk_forward(&n, x);
  /*
  sk_forward(&n, x);
  sk_forward(&n, x);
  printf("two:\n");
  Network two = create_network(test1);
  printf("three:\n");
  Network thr = create_network(test2);
  */
  return 0;
#endif
}
