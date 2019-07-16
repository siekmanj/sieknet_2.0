#include <stdio.h>
#include <stdlib.h>

#include <layer.h>
#include <tensor.h>
#include <sieknet.h>
#include <parser.h>

static void initialize_network(Network *n){
  size_t param_idx = 0;

  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->param_idx  = param_idx;
    sk_initialize_layer(l, n->is_recurrent);
    param_idx += l->num_params;
  }

  Layer *data = (Layer *)malloc(sizeof(Layer));
  data->size       = n->input_dimension;
  data->rank       = -1;
  data->name       = "DATA_IN";
  data->params_per_input = 0;

  if(n->is_recurrent)
    data->output     = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, n->input_dimension);
  else
    data->output     = create_tensor(SIEKNET_CPU, 0, n->input_dimension);

  n->data_layer = data;

  Layer **new_input_layers = (Layer **)malloc((n->layers[0]->num_input_layers + 1) * sizeof(Layer *));

  new_input_layers[0] = data;
  for(int i = 1; i < n->layers[0]->num_input_layers + 1; i++)
    new_input_layers[i] = n->layers[0]->input_layers[i-1];
  free(n->layers[0]->input_layers);
  
  n->layers[0]->input_layers = new_input_layers;
  n->layers[0]->num_input_layers++;

  n->params     = create_tensor(SIEKNET_CPU, param_idx);
  n->param_grad = create_tensor(SIEKNET_CPU, param_idx);

  n->t = 0;
}

Layer *sk_layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    if(!strcmp(n->layers[i]->name, name))
      return n->layers[i];
  }
  return NULL;
}

Network sk_create_network(const char *skfile){
  Network n = {0};

  /* 
   * Retrieve layer names + sizes, input layer names, 
   * logistic functions, layer types, and network name.
   */
  parse_network(&n, skfile);

  /*
   * Construct directed graph, assign input & output layers,
   * determine order of execution and recurrent inputs.
   */
  build_network(&n);

  /*
   *  Allocate memory, initialize layer tensor objects
   */
  initialize_network(&n);

  return n;
}

void sk_forward(Network *n, float *x){

  copy_to_tensor(n->data_layer->output, x, n->t);

  for(int i = 0; i < n->depth; i++)
    sk_layer_forward(n->layers[i], n->params, n->t);

  //tensor_to_arr(n.output, 
}

float sk_cost(Network *n, float *y){
  return 0.0f;
}

void sk_backward(Network *n){

}


