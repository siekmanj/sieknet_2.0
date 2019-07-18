#include <stdio.h>

#include <sieknet.h>
#include <tensor.h>

const char* test0= "model/one.sk";
const char* test1= "model/two.sk";
const char* test2= "model/three.sk";

int main(){
  Tensor a = create_tensor(SIEKNET_CPU, 3, 5, 3, 2, 10);
  print_tensor(a);
  /*
  printf("one:\n");
  Network n = sk_create_network(test0);

  float *x = calloc(n.input_dimension, sizeof(float));
  sk_forward(&n, x);
  sk_forward(&n, x);
  sk_forward(&n, x);
   */
  /*
  printf("two:\n");
  Network two = create_network(test1);
  printf("three:\n");
  Network thr = create_network(test2);
  */
  return 0;
}
