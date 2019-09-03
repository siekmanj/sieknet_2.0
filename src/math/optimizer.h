#ifndef SIEKNET_OPTIMIZER_H
#define SIEKNET_OPTIMIZER_H

#include <stdlib.h>
#include <tensor.h>
#include <stdio.h>

typedef enum optimizer_type_{SK_SGD, SK_MOMENTUM} SK_OPTIMIZER_TYPE;

typedef struct optimizer_{
  Tensor params;
  Tensor gradient;

  void *data;

  float lr;
  float momentum;

  void (*step)(struct optimizer_);

} Optimizer;

Optimizer create_optimizer(Tensor, Tensor, SK_OPTIMIZER_TYPE);
#endif
