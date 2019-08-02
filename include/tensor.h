#ifndef SIEKNET_TENSOR_H
#define SIEKNET_TENSOR_H

#include <conf.h>

typedef enum sk_device_ {SIEKNET_CPU, SIEKNET_GPU} Device;

/*
 * An n-dimensional tensor stored contiguously in memory.
 */
typedef struct tensor_{
  size_t n;
  size_t *dims;
  size_t *strides;

  void *data;

  Device device;
  size_t data_offset;
} Tensor;

#define create_tensor(device, ...) tensor_from_arr(device, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_from_arr(Device, size_t *, size_t);

#define copy_to_tensor(buff, bufflen, tensor, ...) arr_to_tensor(buff, bufflen, tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
void arr_to_tensor(float *, size_t, Tensor, size_t *, size_t);

#define tensor_get_offset(tensor, ...) get_flat_idx(tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
size_t get_flat_idx(Tensor, size_t *, size_t);

#define get_subtensor(tensor, ...) tensor_to_subtensor(tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_to_subtensor(Tensor, size_t *, size_t);


void tensor_to(Device, Tensor *);

float tensor_reduce_dot(const Tensor, const Tensor);

void tensor_mmult(const Tensor, const Tensor, Tensor);


void tensor_elementwise_add(const Tensor, const Tensor, Tensor);

void tensor_elementwise_mult(const Tensor, size_t,
                             const Tensor, size_t,
                             Tensor,       size_t);

void tensor_fill_random(Tensor);
void tensor_zero(Tensor);
void tensor_copy(Tensor, Tensor);

void tensor_sigmoid(Tensor);
void tensor_tanh(Tensor);
void tensor_relu(Tensor);
void tensor_selu(Tensor);
void tensor_softmax(Tensor);

float tensor_quadratic_cost(Tensor, Tensor, Tensor);
float tensor_cross_entropy_cost(Tensor, Tensor, Tensor);

void tensor_print(Tensor);

#endif
