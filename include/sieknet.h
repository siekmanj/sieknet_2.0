#ifndef SIEKNET_NETWORK_H
#define SIEKNET_NETWORK_H

#include <stdlib.h>

#include <conf.h>
#include <layer.h>
#include <tensor.h>

typedef enum sk_cost_fn {SK_QUADRATIC_COST, SK_CROSS_ENTROPY_COST} SK_COST_FN;

typedef struct net_{
  char *name;

  char *input_layername;
  char *output_layername;

  Layer *data_layer;
  Layer *input_layer;
  Layer **layers;
  Layer *output_layer;

  Tensor output;
  
  Tensor params;
  Tensor param_grad;

  size_t trainable;
  size_t input_dimension;
  size_t num_params;
  size_t depth;
  size_t t;

  int is_recurrent;

} Network;

void parse_network(Network *, const char *);
void build_network(Network *);

Network sk_load_network(const char *, const char *);
Network sk_create_network(const char *);
void    sk_save_network(const char *);

void  sk_forward(Network *, Tensor);
float sk_cost(Network *, Layer *, Tensor, SK_COST_FN);

//void  sk_sequence_forward(Network *, Tensor);
//float sk_sequence_cost(SK_COST_FN, Layer *, Tensor);

void  sk_backward(Network *);

Layer *sk_layer_from_name(Network *, const char *);
#endif
