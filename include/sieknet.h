#ifndef SIEKNET_NETWORK_H
#define SIEKNET_NETWORK_H

#include <stdlib.h>

#include <conf.h>
#include <layer.h>
#include <tensor.h>

typedef struct net_{
  char *name;

  char *input_layername;
  char *output_layername;

  Layer *input_layer;
  Layer **layers;
  Layer *output_layer;

  Tensor output;
  
  Tensor params;
  Tensor param_grad;

  size_t input_dimension;
  size_t num_params;
  size_t depth;
  size_t t;

  int is_recurrent;

} Network;

void parse_network(Network *, const char *);
void build_network(Network *);

Network load_network(const char *, const char *);
Network create_network(const char *);
void save_network(const char *);

void  sk_forward(Network *, float *);
float sk_cost(Network *, float *);
void  sk_backward(Network *);

Layer *layer_from_name(Network *, const char *);
#endif
