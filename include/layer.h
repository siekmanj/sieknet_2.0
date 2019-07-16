#ifndef SIEKNET_LAYER_H
#define SIEKNET_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <conf.h>
#include <tensor.h>

typedef enum sk_logistic {SK_SIGMOID, SK_TANH, SK_RELU, SK_LINEAR, SK_SOFTMAX} Logistic;
typedef enum sk_type     {SK_FF, SK_RC, SK_LSTM, SK_GRU, SK_ATT} LayerType;

typedef struct layer_{ 
  char  *name;
  char **input_names;

  struct layer_ **input_layers;
  struct layer_ **output_layers;
  size_t num_input_layers;
  size_t num_output_layers;

  size_t size;
  size_t params_per_input;

  size_t param_idx;
  size_t num_params;

  int rank;
  int visited;

  Tensor input_gradient;
  Tensor gradient;
  Tensor output;
  Tensor loutput;

  LayerType layertype;
  Logistic  logistic;

} Layer;

int contains_layer(Layer **, Layer *, size_t);

void sk_initialize_layer(Layer *, int);

void sk_layer_forward(Layer *);

#endif
