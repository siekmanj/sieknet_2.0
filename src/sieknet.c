#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sieknet.h>

#include <parser.h>

Layer *layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    printf("comparing '%s' to '%s'\n", n->layers[i]->name, name);
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
        printf("ERROR: could not find layer with name '%s' while constructing graph for layer '%s'.\n", l->input_names[j], l->name);
        SK_ERROR("bad graph config");
      }
    }
  }

}
