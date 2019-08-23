#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <layer.h>
#include <parser.h>

typedef struct ntm_data_{
  Tensor bias;
  Tensor *weights;

  Tensor bias_grad;
  Tensor *weight_grad;

  Tensor cell_state;
  Tensor cell_state_tanh;
  Tensor last_cell_state;
  Tensor cell_grad;

  Tensor gates;
  Tensor gate_grads;

  Tensor cell_future_grad;
} LSTM_layer_data;

void sk_ntm_layer_forward(Layer *l, size_t t){}

void sk_ntm_layer_backward(Layer *l, size_t t){}

void sk_ntm_layer_wipe(Layer *l){}

void sk_ntm_layer_parse(Layer *l, char *src){}

void sk_ntm_layer_allocate(Layer *l){}

void sk_ntm_layer_initialize(Layer *l, Tensor p, Tensor g){}
