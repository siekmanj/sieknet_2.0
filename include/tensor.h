#ifndef SIEKNET_TENSOR_H
#define SIEKNET_TENSOR_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <conf.h>

typedef enum sk_device_ {SIEKNET_CPU, SIEKNET_GPU} Device;

/*
 * An n-dimensional tensor stored contiguously in memory.
 */
typedef struct tensor_{
  size_t n;
  size_t *dims;

  void *data;

  Device device;
  size_t data_offset;
} Tensor;

#define create_tensor(device, ...) tensor_from_arr(device, (size_t[]){__VA_ARGS__}, sizeof((size_t[]){__VA_ARGS__})/sizeof(size_t))
Tensor tensor_from_arr(Device, size_t *, size_t);

void send_tensor_to_gpu(Tensor *);
void send_tensor_to_cpu(Tensor *);

#endif
