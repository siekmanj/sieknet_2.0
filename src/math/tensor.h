#ifndef SIEKNET_TENSOR_H
#define SIEKNET_TENSOR_H

#include <conf.h>
#include <math.h>

typedef enum sk_device_ {SIEKNET_CPU, SIEKNET_GPU} TENSOR_DEVICE;
typedef enum sk_tensor_ {TENSOR, SUBTENSOR, RESHAPE} TENSOR_TYPE;

/*
 * An n-dimensional tensor stored contiguously in memory.
 */
typedef struct tensor_{
  size_t n;
  size_t *dims;
  size_t *strides;

  void *data;

  TENSOR_DEVICE device;
  size_t data_offset;
  size_t size;

  TENSOR_TYPE type;
} Tensor;

float uniform(float, float);
float normal(float, float);

#define create_tensor(device, ...) tensor_from_arr(device, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_from_arr(TENSOR_DEVICE, size_t *, size_t);

#define copy_to_tensor(buff, bufflen, tensor, ...) arr_to_tensor(buff, bufflen, tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
void arr_to_tensor(float *, size_t, Tensor, size_t *, size_t);

#define tensor_get_offset(tensor, ...) tensor_flat_idx(tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
size_t tensor_flat_idx(Tensor, size_t *, size_t);

#define get_subtensor(tensor, ...) tensor_to_subtensor(tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_to_subtensor(Tensor, size_t *, size_t); 

#define get_subtensor_reshape(tensor, offset, ...) tensor_to_subtensor_reshape(tensor, offset, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_to_subtensor_reshape(Tensor, size_t, size_t *, size_t);

#define tensor_at(tensor, ...) tensor_at_idx(tensor, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
float tensor_at_idx(Tensor, size_t *, size_t);

int tensor_argmax(Tensor);

void tensor_to(TENSOR_DEVICE, Tensor *);

float tensor_reduce_dot(const Tensor, const Tensor);

void tensor_mmult(const Tensor, const Tensor, Tensor);

void tensor_elementwise_add(const Tensor, const Tensor, Tensor);
void tensor_elementwise_sub(const Tensor, const Tensor, Tensor);
void tensor_elementwise_mul(const Tensor, const Tensor, Tensor);

void tensor_scalar_mul(Tensor, float);

void tensor_transpose(Tensor, size_t, size_t);
void tensor_fill_random(Tensor, float, float);
void tensor_fill(Tensor, float);
void tensor_copy(Tensor, Tensor);

Tensor tensor_clone(TENSOR_DEVICE, Tensor);

void tensor_fabs(Tensor);
void tensor_expf(Tensor);

void tensor_sigmoid_precompute(Tensor, Tensor);
void tensor_tanh_precompute(Tensor, Tensor);
void tensor_relu_precompute(Tensor, Tensor);
void tensor_linear_precompute(Tensor, Tensor);

void tensor_softmax_precompute(Tensor, Tensor);

double tensor_quadratic_cost(Tensor, Tensor, Tensor);
double tensor_cross_entropy_cost(Tensor, Tensor, Tensor);

float *tensor_raw(Tensor);

void tensor_print(Tensor);
void tensor_dealloc(Tensor);

#endif
