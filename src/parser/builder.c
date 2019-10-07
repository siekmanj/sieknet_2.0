#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <parser.h>
#include <sieknet.h>

static int dfs_check_cycle(Layer *origin, Layer *check){
  if(origin == check)
    return 1;

  if(check->visited)
    return 0;
  else
    check->visited = 1;

  for(int i = 0; i < check->num_output_layers; i++)
    if(dfs_check_cycle(origin, check->output_layers[i]))
      return 1;
  return 0;
}

/*
 * Time complexity: O(bad)
 */
void assign_execution_order(Layer **layers, size_t num_layers, Layer *root, size_t rank){
  int num_nonrecurrent_parents = 0;
  int has_parent_cycle = 0;
  int highest_recurrent_parent_rank = -1;
  int highest_nonrecurrent_parent_rank = -1;

  for(int i = 0; i < root->num_input_layers; i++){
    Layer *parent = root->input_layers[i];

    for(int j = 0; j < num_layers; j++)
      layers[j]->visited = 0;
    if(dfs_check_cycle(parent, root)){ // check for a cycle with a parent (meaning recurrent input)
      highest_recurrent_parent_rank = MAX(parent->rank, highest_recurrent_parent_rank);
      has_parent_cycle = 1;
    }else{ // if no cycle with this parent (not recurrent input)
      highest_nonrecurrent_parent_rank = MAX(parent->rank, highest_nonrecurrent_parent_rank);
      if(parent->rank == -1) // if parent has not been assigned execution rank, come back here later (through that parent)
        return; 
      else
        num_nonrecurrent_parents++;
    }
  }

  if(!root->num_input_layers)
    root->rank = 0;
  else if(num_nonrecurrent_parents && has_parent_cycle) // if we have at least one non-recurrent parent and one recurrent parent
    root->rank = MAX(highest_nonrecurrent_parent_rank + 1, highest_recurrent_parent_rank);
  else
    root->rank = rank + 1;

  for(int i = 0; i < root->num_output_layers; i++){
    Layer *child = root->output_layers[i];
    for(int j = 0; j < num_layers; j++)
      layers[j]->visited = 0;

    if(child->rank == -1)
      assign_execution_order(layers, num_layers, child, root->rank);
  }
}

static inline int layer_comparator(const void *a, const void *b){
  int l = (*((Layer**)a))->rank;
  int r = (*((Layer**)b))->rank;
  return l - r;
}

void build_network(Network *n){
 for(int i = 0; i < n->depth; i++)
   n->layers[i]->num_output_layers = 0;

  /*
   * Using the parsed layer names, create a directed graph.
   */
  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->input_layers = calloc(l->num_input_layers, sizeof(Layer*));

    for(int j = 0; j < l->num_input_layers; j++){
      Layer *in = sk_layer_from_name(n, l->input_names[j]);

      if(in){
        l->input_layers[j] = in;
        in->num_output_layers++;
      }
      else
        SK_ERROR("could not find layer with name '%s' while constructing graph for layer '%s'.\n", l->input_names[j], l->name);
    }
  }
  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->output_layers = calloc(l->num_output_layers, sizeof(Layer*));
    l->num_output_layers = 0;
    l->rank = -1;
  }

  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    for(int j = 0; j < l->num_input_layers; j++){
      Layer *in = l->input_layers[j];
      in->output_layers[in->num_output_layers++] = l;
    }
  }

  n->input_layer  = sk_layer_from_name(n, n->input_layername);

  if(!n->input_layer)
    SK_ERROR("could not find a layer with name '%s' while searching for network input layer.", n->input_layername);

  assign_execution_order(n->layers, n->depth, n->input_layer, 0);
  qsort((void*)n->layers, n->depth, sizeof(Layer*), layer_comparator);

  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->idx = i;
    for(int j = 0; j < l->num_output_layers; j++){
      Layer *out = l->output_layers[j];
      if(out->rank <= l->rank){
        l->output_recurs = 1;
        n->is_recurrent = 1;
      }
    }
  }

  int chunks[n->depth+1];
  int num_chunks = 0;

  int tentative_chunk_boundary = n->depth-1;
  for(int i = n->depth-1; i >= 0; i--){
    Layer *l = n->layers[i];
    for(int j = 0; j < l->num_output_layers; j++){
      Layer *out = l->output_layers[j];
      if(out->rank <= l->rank){
        tentative_chunk_boundary = out->idx;
      }
    }
    if(i == tentative_chunk_boundary){
      chunks[num_chunks++] = i;
      tentative_chunk_boundary = i-1;
    }
  }

#if 0
  for(int i = 0; i < num_chunks; i++){
    printf("Doing chunk %d: %d and %d\n", i, chunks[i]);
    printf("Chunk between %d '%s'\n", chunks[i], n->layers[chunks[i]]->name);
  }
#endif

  n->chunks = malloc(sizeof(size_t) * num_chunks);
  n->num_chunks = num_chunks;
  for(int i = num_chunks-1; i >= 0; i--){
    n->chunks[i] = chunks[num_chunks-i-1];
  }
}
