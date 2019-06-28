#ifndef SIEKNET_LAYER_H
#define SIEKNET_LAYER_H

#include <stdlib.h>
#include <conf.h>

typedef enum sk_type     {SK_FF, SK_RC, SK_LSTM, SK_ATT} LayerType;
typedef enum sk_logistic {SK_SIGMOID, SK_TANH, SK_RELU, SK_SOFTMAX} Logistic;

typedef struct layer_{ 
  char  *name;

  struct layer_ **input_layers;
  size_t num_input_layers;
  size_t size;
  size_t param_idx;
  size_t real_idx;
  size_t params_per_input;
  size_t num_params;
  size_t num_reals;

  void  *data;
  float *output;

  LayerType layertype;
  Logistic  logistic;

} Layer;

typedef struct net_{
  char *name;

  Layer *input_layer;
  Layer **layers;
  Layer *output_layer;
  
  float *params;
  float *reals;

  size_t num_params;
  size_t num_reals;
  size_t depth;
  size_t t;

} Network;

Network load_network(const char *, const char *);
Network create_network(const char *);
void save_network(const char *);

void  sk_forward(Network *, float *);
float sk_cost(float *, float *, float *, size_t);
void  sk_backward(Network *);

#define sk_err(x) \
  do {                                                             \
    printf("ERROR: in file %s:%d, '%s'\n", __FILE__, __LINE__, x); \
    exit(1);                                                       \
  } while(0);

#endif
