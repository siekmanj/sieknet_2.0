#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <parser.h>

typedef struct Softmax_layer_data_{

	Tensor softmax_jacobian;
	Tensor input_gradient;

} SM_layer_data;

/*
 * Computes the forward pass for a layer for a single 
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_softmax_layer_forward(Layer *l, size_t t){
	SM_layer_data *d = (SM_layer_data*)l->data;

  Tensor logits = get_subtensor(l->output, t);
	Tensor jacobian = get_subtensor(d->softmax_jacobian, t);

  /* Zero the output tensor for this timestep */
  tensor_fill(logits, 0.0f);
	tensor_fill(jacobian, 0.0f);

  size_t logit_offset = 0;

  /* Loop through all the input layers and copy to logits */
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    /* Get the subtensor for this timestep */
    Tensor x = in->rank >= l->rank ? in->loutput : get_subtensor(in->output, t);

		Tensor logit_x = get_subtensor_reshape(logits, logit_offset, x.size);

		tensor_copy(x, logit_x);
		tensor_dealloc(logit_x);
		logit_offset += x.size;
  }
	tensor_softmax_precompute(logits, jacobian);
}

/*
 * Computes the backward pass for a layer for a single
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_softmax_layer_backward(Layer *l, size_t t){
	SM_layer_data *d = (SM_layer_data*)l->data;

  Tensor gradient = get_subtensor(l->gradient, t);
	Tensor jacobian = get_subtensor(d->softmax_jacobian, t);
	Tensor input_grad = get_subtensor(d->input_gradient, t);

	tensor_fill(input_grad, 0.0f);
	tensor_mmult(jacobian, gradient, input_grad);

	size_t logit_offset = 0;
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    int target_t = in->rank >= l->rank ? t - 1 : t;

    /* Get the subtensor for this timestep */
    if(target_t >= 0 && in->gradient.data){
      Tensor dx = get_subtensor(in->gradient, target_t);

			Tensor logit_gradient = get_subtensor_reshape(input_grad, logit_offset, dx.size);

			tensor_elementwise_add(logit_gradient, dx, dx);
			tensor_dealloc(logit_gradient);
      
			logit_offset += in->size;
    }else{
			logit_offset += in->size;
			continue;
		}
  }
}

void sk_softmax_layer_dealloc(Layer *l){
  tensor_dealloc(l->output);
  tensor_dealloc(l->gradient);
  tensor_dealloc(l->loutput);

  SM_layer_data *d = (SM_layer_data*)l->data;
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

  tensor_dealloc(d->softmax_jacobian);
  tensor_dealloc(d->input_gradient);

  free(d);
  free(l->name);
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
void sk_softmax_layer_count_params(Layer *l){
  l->num_params = 0;
  l->num_consts = 0;
}

/*
 * Allocates things
 */
void sk_softmax_layer_initialize(Layer *l){
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
  l->dealloc      = sk_softmax_layer_dealloc;

	SM_layer_data *d = calloc(sizeof(SM_layer_data), 1);
	d->softmax_jacobian = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size, l->size);
	d->input_gradient   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
	l->data = d;
}


