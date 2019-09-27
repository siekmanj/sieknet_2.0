#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <parser.h>

void sk_identity_layer_forward(Layer *l, size_t t){
  Layer *in = l->input_layers[0];

  if(in->rank >= l->rank)
    SK_ERROR("Cannot have recurrent connection to identity layer (Input rank %d, layer rank %d)", in->rank, l->rank);

  tensor_copy(get_subtensor(in->output, t), get_subtensor(l->output, t));
}

void sk_identity_layer_backward(Layer *l, size_t t){
  Layer *in = l->input_layers[0];
  if(in->gradient.data)
    tensor_copy(get_subtensor(l->gradient, t), get_subtensor(in->gradient, t));
}

void sk_identity_layer_count_size(Layer *l){
  l->size = l->input_layers[0]->size;
  l->num_params = 0;
  l->num_consts = 0;
}

void sk_identity_layer_dealloc(Layer *l){
  tensor_dealloc(l->output);
  tensor_dealloc(l->gradient);
  tensor_dealloc(l->loutput);

  for(int i = 0; i < l->num_input_layers; i++){
    if(l->input_names){
      free(l->input_names[i]);
    }
  }
  if(l->input_layers)
    free(l->input_layers);

  if(l->output_layers)
    free(l->output_layers);

  if(l->input_names)
    free(l->input_names);

  free(l->name);
}

void sk_identity_layer_wipe(Layer *l){};

/*
 * Parses the attributes of a identity layer from
 * an excerpt of a config file.
 */
void sk_identity_layer_parse(Layer *l, char *src){

  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse identity-layer attribute 'name'.");

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  if(num_names > 1)
    SK_ERROR("Identity layers can only have one input!");

  l->input_names = input_names;
  l->num_input_layers = num_names;
  l->name = name;
}

void sk_identity_layer_initialize(Layer *l){
  if(l->num_input_layers > 1)
    SK_ERROR("Cannot have more than one input layer.\n");

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->loutput        = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_identity_layer_forward;
  l->backward     = sk_identity_layer_backward;
  l->wipe         = sk_identity_layer_wipe;
  l->dealloc      = sk_identity_layer_dealloc;
}
