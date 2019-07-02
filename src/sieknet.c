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

int contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}

int dfs_check_cycle(Layer *origin, Layer *check){
  printf("checking for cycle between '%s' and '%s'\n", origin->name, check->name);
  if(origin == check)
    return 1;

  for(int i = 0; i < check->num_output_layers; i++){
    Layer *output = check->output_layers[i];
    printf("  dfs_check_cycle(): comparing child '%s' to root '%s'\n", output->name, origin->name);
    if(dfs_check_cycle(origin, output))
      return 1;
  }
  return 0;
}

void assign_execution_order(Layer *current, size_t rank){
  for(int i = 0; i < current->num_input_layers; i++){
    Layer *parent = current->input_layers[i];
    //if(parent->rank == -1){
      if(parent->layertype == SK_RC){
        if(dfs_check_cycle(parent, current)){
          printf("Parent '%s' has cycle with '%s'- continuing.\n", parent->name, current->name);
          continue;
        }else
          printf("Parent '%s' is recurrent but no cycle found - holding off\n", parent->name);
      }
      else if(parent->rank == -1){
        printf("could not assign rank to '%s' yet, parent missing\n", current->name);
        return;
      }
  }

  current->rank = rank;
  printf("'%s' now has rank %lu\n", current->name, current->rank);
  getchar();

  for(int i = 0; i < current->num_output_layers; i++){
    Layer *child = current->output_layers[i];
    printf("checking child '%s' (parent '%s', rank %lu)\n", child->name, current->name, current->rank);
    if(child->rank == -1)
      assign_execution_order(child, current->rank+1);
    //else
    //  if(current->layertype != SK_RC)
    //    SK_ERROR("found a cycle between '%s' and '%s', but '%s' was not declared recurrent.", current->name, child->name, child->name);
  }
}


Network create_network(const char *skfile){

  /* 
   * Retrieve layer names + sizes, input layer names, 
   * logistic functions, layer types, and network name.
   */
  Network n = parse_network(skfile);

 for(int i = 0; i < n.depth; i++)
   n.layers[i]->num_output_layers = 0;

  /*
   * Using the parsed layer names, create a directed graph.
   */
  for(int i = 0; i < n.depth; i++){
    Layer *l = n.layers[i];
    l->input_layers = calloc(l->num_input_layers, sizeof(Layer*));

    for(int j = 0; j < l->num_input_layers; j++){
      Layer *in = layer_from_name(&n, l->input_names[j]);
      in->num_output_layers++;

      if(in)
        l->input_layers[j] = in;
      else
        SK_ERROR("could not find layer with name '%s' while constructing graph for layer '%s'.\n", l->input_names[j], l->name);
    }
  }
  for(int i = 0; i < n.depth; i++){
    Layer *l = n.layers[i];
    l->output_layers = calloc(l->num_output_layers, sizeof(Layer*));
    l->num_output_layers = 0;
    l->rank = -1;
  }

  for(int i = 0; i < n.depth; i++){
    Layer *l = n.layers[i];
    for(int j = 0; j < l->num_input_layers; j++){
      Layer *in = l->input_layers[j];
      in->output_layers[in->num_output_layers++] = l;
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

  assign_execution_order(n.input_layer, 0);
  for(int i = 0; i < n.depth; i++)
    printf("'%s' rank: %lu\n", n.layers[i]->name, n.layers[i]->rank);
  
}

