#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sieknet.h>

#include <parser.h>

Layer *layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    if(!strcmp(n->layers[i]->name, name))
      return n->layers[i];
  }
  return NULL;
}

Network create_network(const char *skfile){

  /* 
   * Retrieve layer names + sizes, input layer names, 
   * logistic functions, layer types, and network name.
   */
  Network n = parse_network(skfile);

  /*
   * Using the parsed layer names, create a directed graph.
   */
   for(int i = 0; i < n.depth; i++){
    Layer *l = n.layers[i];
    l->input_layers = calloc(l->num_input_layers, sizeof(Layer*));

    for(int j = 0; j < l->num_input_layers; j++){
      Layer *in = layer_from_name(&n, l->input_names[j]);
      if(in){
        l->input_layers[j] = in;
        printf("'%s' now connected to '%s'\n", l->name, l->input_layers[j]->name);
      }else{
        SK_ERROR("could not find layer with name '%s' while constructing graph for layer '%s'.\n", l->input_names[j], l->name);
      }
    }
  }

  n.input_layer  = layer_from_name(&n, n.input_layername);
  n.output_layer = layer_from_name(&n, n.output_layername);

  if(!n.input_layer)
    SK_ERROR("could not find a layer with name '%s' while searching for network input layer.", n.input_layername);

  if(!n.output_layer)
    SK_ERROR("could not find a layer with name '%s' while searching for network output layer.", n.output_layername);

  Layer *execution_order[n.depth];
  memset(execution_order, '\0', sizeof(Layer*) * n.depth);
  size_t idx = 0;

  execution_order[0]         = n.input_layer;
  execution_order[n.depth-1] = n.output_layer;
  
  Layer *current = n.output_layer;
  while(current){

  }

}
