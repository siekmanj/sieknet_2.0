#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <layer.h>
#include <parser.h>


typedef struct fc_data_{
  Tensor bias;
  Tensor *weights;
  Tensor bias_grad;
  Tensor *weight_grad;
  Tensor intermediate_grad;
} FC_layer_data;

/*
 * Computes the forward pass for a layer for a single 
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_fc_layer_forward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor b = d->bias;
  Tensor y = get_subtensor(l->output, t);
  Tensor dy = get_subtensor(d->intermediate_grad, t);

  d->intermediate_grad.dims[0] = t + 1;

  /* Zero the output tensor for this timestep */
  tensor_fill(y, 0.0f);

  /* Loop through all the input layers and do a matrix mult */
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    Tensor w = d->weights[i];

    /* Get the subtensor for this timestep */
    Tensor x = in->rank >= l->rank ? in->loutput : in->output;
    x        = x.n == 1 ? x : get_subtensor(x, t);

    /* Matrix multiplication between weights and input */
    tensor_transpose(w, 0, 1);
    tensor_mmult(w, x, y);
    tensor_transpose(w, 0, 1);
  }
	/* Elementwise-add the bias to the output */
  tensor_elementwise_add(b, y, y);
  l->nonlinearity(y, dy);
}

/*
 * Computes the backward pass for a layer for a single
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_fc_layer_backward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor o = get_subtensor(l->gradient, t);
  Tensor g = get_subtensor(d->intermediate_grad, t);
  tensor_elementwise_mul(o, g, g); 

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    Tensor w  = d->weights[i];
    Tensor dw = d->weight_grad[i];

    Tensor x  = {0};
    Tensor dx = {0};
    if(in->rank >= l->rank){ // If this is a recurrent connection
      if(t > 0){             // If there exists a t-1th output
        x = get_subtensor(in->output, t-1);
        if(in->gradient.data)
          dx = get_subtensor(in->gradient, t-1);
      }else continue;
    }else{
      x = get_subtensor(in->output, t);
      if(in->gradient.data)
        dx = get_subtensor(in->gradient, t);
    }

    /* Compute weight gradients */
    tensor_mmult(x, g, dw); // dW = x * g

    /* Compute input gradients if needed */
    if(dx.data)
      tensor_mmult(w, g, dx); // dX = g * w
  }

  /* Compute bias gradients */
  Tensor db = d->bias_grad;
  tensor_elementwise_add(g, db, db);
}

/*
 * Allocates the memory for a fully-connected layer.
 */
void sk_fc_layer_allocate(Layer *l){
  l->num_params = 0;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += (l->size * in->size) + l->size;
  }

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  l->loutput      = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_fc_layer_forward;
  l->backward     = sk_fc_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(l->logistic);

  FC_layer_data *d     = calloc(1, sizeof(FC_layer_data));
  d->weights           = calloc(l->num_input_layers, sizeof(Tensor));
  d->weight_grad       = calloc(l->num_input_layers, sizeof(Tensor));
  d->intermediate_grad = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->data = d;
}

/*
 * Parses the attributes of a fully-connected layer from
 * an excerpt of a config file.
 */
void sk_fc_layer_parse(Layer *l, char *src){

  int size;
  sk_parser_find_int("size", src, &size);

  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse fc-layer attribute 'name'.");

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  SK_LOGISTIC logistic;
  char *logistic_src;
  if(sk_parser_find_string("logistic", src, &logistic_src))
    logistic = sk_layer_parse_logistic(logistic_src);
  else
    logistic = SK_SIGMOID;
  free(logistic_src);

  l->size = size;
  l->input_names = input_names;
  l->logistic = logistic;
  l->num_input_layers = num_names;
  l->name = name;
}

/*
 * Performs weight initialization, subtensoring from network
 * parameter/parameter gradient tensors.
 */
void sk_fc_layer_initialize(Layer *l, Tensor p, Tensor g){
  /* 
   * Perform weight initialization according to the desired scheme. 
   * Default is Xavier initialization.
   */
  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;
  switch(l->weight_initialization){
    case SK_XAVIER:{
      float *theta = &tensor_raw(p)[l->param_idx];
      for(int i = 0; i < l->num_params; i++){
        theta[i] = normal(0, 1 / sqrt(input_dim));
      }
      break;
    }
    case SK_HE:{
      float *theta = &tensor_raw(p)[l->param_idx];
      for(int i = 0; i < l->num_params; i++)
        theta[i] = normal(0, sqrt(2 / input_dim));
      break;
    }
    default:{
      SK_ERROR("Not implemented!");
      break;
    }
  }

  /*
   * Set up weights and biases of this layer. We will use an internal struct
   * which is not used anywhere outside of this file (FC_layer_data). This 
   * is used to manage the forward and backward passes for fully connected
   * layers.
   */
  size_t param_offset = l->param_idx;
  FC_layer_data *d = (FC_layer_data *)l->data;

  d->bias      = get_subtensor_reshape(p, param_offset, l->size);
  d->bias_grad = get_subtensor_reshape(g, param_offset, l->size);

  param_offset += l->size;

  for(int i = 0; i < l->num_input_layers; i++){

    d->weights[i]     = get_subtensor_reshape(p, param_offset, l->input_layers[i]->size, l->size);
    d->weight_grad[i] = get_subtensor_reshape(g, param_offset, l->input_layers[i]->size, l->size);

    param_offset += l->size * l->input_layers[i]->size;
  }

  l->output.dims[0] = 1;
  l->gradient.dims[0] = 1;
}