#ifndef SIEKNET_LAYER_H
#define SIEKNET_LAYER_H

#include <stdlib.h>
#include <conf.h>

typedef enum sk_type     {SK_FF, SK_RC, SK_LSTM, SK_ATT} LayerType;
typedef enum sk_logistic {SK_SIGMOID, SK_TANH, SK_RELU, SK_LINEAR, SK_SOFTMAX} Logistic;

typedef struct tensor_{
  size_t n;
  size_t *dims;
  void   *data;
} Tensor;


typedef struct layer_{ 
  char  *name;
  char **input_names;

  struct layer_ **input_layers;
  struct layer_ **output_layers;
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

  char *input_layername;
  char *output_layername;

  Layer *input_layer;
  Layer **layers;
  Layer *output_layer;
  
  float *params;
  float *reals;

  size_t input_dimension;
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

#define SK_ERROR(x) \
  do {                                                             \
    printf("ERROR: in file %s:%d, '%s'\n", __FILE__, __LINE__, x); \
    exit(1);                                                       \
  } while(0)

#define STATIC_LEN(arr) (sizeof(arr) / sizeof(arr[0]))
#endif
