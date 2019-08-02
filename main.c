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
  Tensor a = create_tensor(SIEKNET_CPU, 4);
  Tensor b = create_tensor(SIEKNET_CPU, 4);
  Tensor c = create_tensor(SIEKNET_CPU, 4);
  tensor_fill_random(a);
  tensor_fill_random(b);
  tensor_print(a);
  tensor_print(b);
  tensor_elementwise_add(a, b, c);
  tensor_print(c);
#endif
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
#if 1
  Network n = sk_create_network(test0);

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

  #if 0
  sk_forward(&n, x1);
  float c1 = sk_cost(&n, n.layers[1], y1, SK_QUADRATIC_COST);
  sk_forward(&n, x2);
  float c2 = sk_cost(&n, n.layers[1], y2, SK_QUADRATIC_COST);
  sk_forward(&n, x3);
  float c3 = sk_cost(&n, n.layers[1], y3, SK_QUADRATIC_COST);

  printf("cost 1: %f\n", c1 + c2 + c3);
#endif

  Tensor x = create_tensor(SIEKNET_CPU, 3, n.input_dimension);
  Tensor y = create_tensor(SIEKNET_CPU, 3, n.layers[1]->size);
  tensor_fill_random(x);
  tensor_fill_random(y);
  sk_forward(&n, x);
  printf("cost 2: %f\n", sk_cost(&n, n.layers[1], y, SK_QUADRATIC_COST));
  tensor_print(n.layers[1]->gradient);
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
#endif
}
