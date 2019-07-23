#include <stdio.h>

#include <sieknet.h>
#include <tensor.h>
#include <util.h>

const char* test0= "model/one.sk";
const char* test1= "model/two.sk";
const char* test2= "model/three.sk";

int main(){
  srand(0);
#if 1

  Tensor a = create_tensor(SIEKNET_CPU, 2, 1);
  tensor_fill_random(a);
  tensor_print(a);
  
  Tensor b = create_tensor(SIEKNET_CPU, 3, 2);
  tensor_fill_random(b);
  tensor_print(b);

  Tensor c = create_tensor(SIEKNET_CPU, 1, 2);
  tensor_print(c);

  tensor_mmult(a, 1, 0,
               b, 1, 0,
               c, 1, 0);

  tensor_print(c);

#else
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
