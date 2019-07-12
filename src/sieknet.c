#include <stdio.h>
#include <stdlib.h>

#include <layer.h>
#include <tensor.h>
#include <sieknet.h>
#include <parser.h>


Layer *layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    if(!strcmp(n->layers[i]->name, name))
      return n->layers[i];
  }
  return NULL;
}

void initialize_network(Network *n){
  size_t param_idx = 0;
  size_t real_idx  = 0;

  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->param_idx  = param_idx;
    initialize_layer(l, n->is_recurrent);
    param_idx += l->num_params;
  }
  //n->params     = c
  //n->param_grad = 
  n->t = 0;
}

Network create_network(const char *skfile){
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
   *  Allocate memory, initialize layer tensor objects,
   */
  initialize_network(&n);

  return n;
}

void sk_forward(Network *n, float *x){

}

float sk_cost(Network *n, float *y){
  return 0.0f;
}

void sk_backward(Network *n){

}


