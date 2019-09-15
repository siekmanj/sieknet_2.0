#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <parser.h>

/*
 * Constants:
 * last mean (layer size)
 * last mean diff (layer size)
 * num steps
 */

typedef struct welford_layer_data_{

  size_t num_steps;
  Tensor steps;
  Tensor mean;
  Tensor std;

  Tensor mean_diff;
  Tensor last_mean;
  Tensor last_mean_diff;

  Tensor intermediate_result;

} WF_layer_data;

/*
 * Computes the forward pass for a layer for a single 
 * timestep.
 */
void sk_welford_layer_forward(Layer *l, size_t t){
  WF_layer_data *d = (WF_layer_data*)l->data;

  if(l->input_layers[0]->rank >= l->rank)
    SK_ERROR("Cannot do recurrent connections with welford layer\n");

  Tensor x = get_subtensor(l->input_layers[0]->output, t);

#if 1
  Tensor mu = get_subtensor(d->mean, t);
  Tensor md = get_subtensor(d->mean_diff, t);
  //if(l->trainable){
    if(d->num_steps == 1){
      tensor_copy(x, d->mean);
      tensor_fill(d->mean_diff, 1e-2);
      tensor_fill(d->std, 1.0f);
    }else{
      tensor_elementwise_sub(x, d->last_mean, mu);
      tensor_scalar_mul(x, 1/(d->num_steps), x);
      tensor_elementwise_add(mu, d->last_mean, mu);

      tensor_elementwise_sub(x, d->last_mean, md);
      tensor_elementwise_sub(x, mu, d->intermediate_result);
      tensor_elementwise_mul(md, d->intermediate_result, d->intermediate_result);
      tensor_elementwise_add(d->last_mean_diff, md, md);
    }
    d->num_steps++;
    tensor_copy(mu, d->last_mean);
    tensor_copy(md, d->last_mean_diff);
  //}

  Tensor std = get_subtensor(d->std, t);
  //tensor_
#endif

  
}

/*
 * Computes the backward pass for a layer for a single
 * timestep.
 */
void sk_welford_layer_backward(Layer *l, size_t t){

}

void sk_welford_layer_dealloc(Layer *l){
  tensor_dealloc(l->output);
  tensor_dealloc(l->gradient);
  tensor_dealloc(l->loutput);
  if(l->input_layers)
    free(l->input_layers);

  if(l->output_layers)
    free(l->output_layers);

  if(l->input_names)
    free(l->input_names);

  //tensor_dealloc(d->welford_jacobian);
  //tensor_dealloc(d->input_gradient);

  free(l->data);
  free(l->name);
}

void sk_welford_layer_wipe(Layer *l){};

/*
 * Parses the attributes of a fully-connected layer from
 * an excerpt of a config file.
 */
void sk_welford_layer_parse(Layer *l, char *src){

  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse welford-layer attribute 'name'.");

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  if(num_names > 1)
    SK_ERROR("Cannot concatenate inputs into welford layer!\n");

  l->input_names = input_names;
  l->num_input_layers = num_names;
  l->name = name;
}

/*
 * Allocates the memory for a fully-connected layer.
 */
void sk_welford_layer_count_params(Layer *l){
  l->num_params = 0;
  l->num_consts = 0;
}

/*
 * Allocates things
 */
void sk_welford_layer_initialize(Layer *l){
  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;

  l->size = input_dim;

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  l->loutput      = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_welford_layer_forward;
  l->backward     = sk_welford_layer_backward;
  l->wipe         = sk_welford_layer_wipe;
  l->dealloc      = sk_welford_layer_dealloc;

}



