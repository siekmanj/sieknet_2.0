#ifndef SIEKNET_NETWORK_H
#define SIEKNET_NETWORK_H

#include <stdlib.h>

#include <conf.h>
#include <tensor.h>

typedef enum sk_cost_fn   {SK_QUADRATIC_COST, SK_CROSS_ENTROPY_COST} SK_COST_FN;
typedef enum sk_logistic  {SK_SIGMOID, SK_TANH, SK_RELU, SK_LINEAR} SK_LOGISTIC;
typedef enum sk_init_type {SK_XAVIER, SK_HE} SK_INIT_TYPE;
typedef enum sk_type      {SK_ID, SK_FF, SK_LSTM, SK_SOFTMAX, SK_NTM, SK_GRU, SK_ATT} SK_LAYER_TYPE;

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

  size_t const_idx;
  size_t num_consts;

  int rank;
  int visited;

  Tensor gradient;
  Tensor output;
  Tensor loutput;

  int blocking;
  int frozen;

  void *data;

  SK_LAYER_TYPE layertype;
  SK_LOGISTIC logistic;
  SK_INIT_TYPE weight_initialization;

  void (*forward)(struct layer_*, size_t);
  void (*backward)(struct layer_*, size_t);
  void (*nonlinearity)(Tensor, Tensor);
  void (*wipe)(struct layer_*);
  void (*dealloc)(struct layer_*);

} Layer;

typedef struct net_{
  char *name;

  char *input_layername;
  char *output_layername;

  Layer *data_layer;
  Layer *input_layer;
  Layer **layers;
  Layer *output_layer;

  Tensor params;
  Tensor param_grad;

  Tensor constants;

  size_t trainable;
  size_t input_dimension;
  size_t num_params;
  size_t num_consts;
  size_t depth;
  size_t t;

  int is_recurrent;
  int is_seq2seq;

} Network;

typedef void (*SK_LOGISTIC_FN)(Tensor);
void         (*sk_logistic_to_fn(SK_LOGISTIC))(Tensor, Tensor);

SK_LAYER_TYPE sk_layer_parse_identifier(const char *);
SK_LOGISTIC   sk_layer_parse_logistic(const char *);

void parse_network(Network *, char *);
void build_network(Network *);

Network sk_create(const char *);
Network sk_load(const char *, const char *);
void    sk_save(Network *, const char *);

void   sk_forward(Network *, Tensor);
double sk_cost(Layer *, Tensor, SK_COST_FN);
void   sk_backward(Network *);
void   sk_wipe(Network *);
void   sk_dealloc(Network *);

void sk_layer_parse(Layer *, char *);
void sk_layer_count_params(Layer *);
void sk_layer_initialize(Layer *, Tensor, Tensor);

int contains_layer(Layer **, Layer *, size_t);
Layer *sk_layer_from_name(Network *, const char *);
#endif
