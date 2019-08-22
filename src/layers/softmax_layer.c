#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <layer.h>
#include <parser.h>

typedef struct Softmax_layer_data_{
  Tensor *input_mappings

} SM_layer_data;

/*
 * Computes the forward pass for a layer for a single 
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_softmax_layer_forward(Layer *l, size_t t){
  Tensor logits = get_subtensor(l->output, t);

  /* Zero the output tensor for this timestep */
  tensor_fill(logits, 0.0f);

  size_t logit_offset = 0;
  /* Loop through all the input layers and do a matrix mult */
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    /* Get the subtensor for this timestep */
    Tensor x = in->rank >= l->rank ? in->loutput : in->output;
    x        = x.n == 1 ? x : get_subtensor(x, t);

  }
}

/*
 * Computes the backward pass for a layer for a single
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_softmax_layer_backward(Layer *l, size_t t){
  Tensor o = get_subtensor(l->gradient, t);

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    int target_t = in->rank >= l->rank ? t - 1 : t;

    /* Get the subtensor for this timestep */
    if(target_t >= 0){
      Tensor dx = {0};
      Tensor x = get_subtensor(in->output, target_t);

      if(in->gradient.data)
        dx = get_subtensor(in->gradient, target_t);
      
    }else continue;

  }
}

void sk_softmax_layer_wipe(Layer *l){};

/*
 * Parses the attributes of a fully-connected layer from
 * an excerpt of a config file.
 */
void sk_softmax_layer_parse(Layer *l, char *src){

  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse softmax-layer attribute 'name'.");

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  l->input_names = input_names;
  l->num_input_layers = num_names;
  l->name = name;
}

/*
 * Allocates the memory for a fully-connected layer.
 */
void sk_softmax_layer_allocate(Layer *l){
  l->num_params = 0;

  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;

  l->size = input_dim;

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  l->loutput      = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_softmax_layer_forward;
  l->backward     = sk_softmax_layer_backward;
  l->wipe         = sk_softmax_layer_wipe;
}

/*
 * Doesn't do a damn thing.
 */
void sk_softmax_layer_initialize(Layer *l, Tensor p, Tensor g){}


