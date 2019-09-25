#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <parser.h>

typedef struct fc_data_{
  Tensor bias;
  Tensor *weights;
  Tensor bias_grad;
  Tensor *weight_grad;
  Tensor intermediate_grad;
  Tensor activation_grad;
} FC_layer_data;

/*
 * Computes the forward pass for a layer for a single 
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_fc_layer_forward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor y = get_subtensor(l->output, t);
  Tensor dy = get_subtensor(d->activation_grad, t);

  d->intermediate_grad.dims[0] = t + 1;
  d->activation_grad.dims[0] = t + 1;

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
  tensor_elementwise_add(d->bias, y, y);
  l->nonlinearity(y, dy);
}

/*
 * Computes the backward pass for a layer for a single
 * timestep. Supports both recurrent and nonrecurrent
 * connections.
 */
void sk_fc_layer_backward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor o  = get_subtensor(l->gradient, t);
  Tensor dy = get_subtensor(d->activation_grad, t);
  Tensor g  = get_subtensor(d->intermediate_grad, t);
  tensor_elementwise_mul(o, dy, g); 

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    Tensor w  = d->weights[i];
    Tensor dw = d->weight_grad[i];

    Tensor x  = {0};
    Tensor dx = {0};

    int target_t = in->rank >= l->rank ? t - 1 : t;

    if(target_t >= 0){
      x = get_subtensor(in->output, target_t);

      if(in->gradient.data)
        dx = get_subtensor(in->gradient, target_t);
      
    }else continue;

    /* Compute weight gradients */
    if(!l->frozen)
      tensor_mmult(x, g, dw); // dW = x * g

    /* Compute input gradients if needed */
    if(!l->blocking && dx.data)
      tensor_mmult(w, g, dx); // dX = g * w
  }

  /* Compute bias gradients */
  if(!l->frozen){
    Tensor db = d->bias_grad;
    tensor_elementwise_add(g, db, db);
  }
}

void sk_fc_layer_dealloc(Layer *l){
  tensor_dealloc(l->output);
  tensor_dealloc(l->gradient);
  tensor_dealloc(l->loutput);

  FC_layer_data *d = (FC_layer_data*)l->data;
  for(int i = 0; i < l->num_input_layers; i++){
    if(l->input_names){
      free(l->input_names[i]);
    }
    tensor_dealloc(d->weights[i]);
    tensor_dealloc(d->weight_grad[i]);

  }
  free(d->weights);
  free(d->weight_grad);

  if(l->input_layers)
    free(l->input_layers);

  if(l->output_layers)
    free(l->output_layers);

  if(l->input_names)
    free(l->input_names);

  tensor_dealloc(d->intermediate_grad);
  tensor_dealloc(d->activation_grad);
  tensor_dealloc(d->bias);
  tensor_dealloc(d->bias_grad);

  free(d);
  free(l->name);
}


void sk_fc_layer_wipe(Layer *l){};

/*
 * Parses the attributes of a fully-connected layer from
 * an excerpt of a config file.
 */
void sk_fc_layer_parse(Layer *l, char *src){

  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse fc-layer attribute 'name'.");

  int size;
  if(!sk_parser_find_int("size", src, &size))
    SK_ERROR("Unable to parse fc-layer attribute 'size' for layer '%s'.\n", name);

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  SK_LOGISTIC logistic;
  char *logistic_src = NULL;
  if(sk_parser_find_string("logistic", src, &logistic_src))
    logistic = sk_layer_parse_logistic(logistic_src);
  else
    logistic = SK_SIGMOID;
  if(logistic_src)
    free(logistic_src);

  l->size = size;
  l->input_names = input_names;
  l->logistic = logistic;
  l->num_input_layers = num_names;
  l->name = name;
}

/*
 * Allocates the memory for a fully-connected layer.
 */
void sk_fc_layer_count_params(Layer *l){
  l->num_params = 0;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += (l->size * in->size);
  }
  l->num_params += l->size;
  l->num_consts = 0;
}

/*
 * Performs weight initialization, subtensoring from network
 * parameter/parameter gradient tensors, allocates memory
 */
void sk_fc_layer_initialize(Layer *l, Tensor p, Tensor g){
  l->blocking = 0;
  l->frozen   = 0;

  l->output   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->loutput  = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_fc_layer_forward;
  l->backward     = sk_fc_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(l->logistic);
  l->wipe         = sk_fc_layer_wipe;
  l->dealloc      = sk_fc_layer_dealloc;

  FC_layer_data *d     = calloc(1, sizeof(FC_layer_data));
  d->weights           = calloc(l->num_input_layers, sizeof(Tensor));
  d->weight_grad       = calloc(l->num_input_layers, sizeof(Tensor));
  d->intermediate_grad = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  d->activation_grad   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;

  /*
   * Set up weights and biases of this layer. We will use an internal struct
   * which is not used anywhere outside of this file (FC_layer_data). This 
   * is used to manage the forward and backward passes for fully connected
   * layers.
   */
  size_t param_offset = l->param_idx;

  d->bias      = get_subtensor_reshape(p, param_offset, l->size);
  d->bias_grad = get_subtensor_reshape(g, param_offset, l->size);

  param_offset += l->size;

  for(int i = 0; i < l->num_input_layers; i++){

    d->weights[i]     = get_subtensor_reshape(p, param_offset, l->input_layers[i]->size, l->size);
    d->weight_grad[i] = get_subtensor_reshape(g, param_offset, l->input_layers[i]->size, l->size);

    if(l->weight_initialization == SK_XAVIER)
        tensor_fill_random(d->weights[i], 0, 1 / sqrt(input_dim));
    if(l->weight_initialization == SK_HE)
        tensor_fill_random(d->weights[i], 0, sqrt(2 / input_dim));

    param_offset += l->size * l->input_layers[i]->size;
  }

  if(l->weight_initialization == SK_XAVIER)
      tensor_fill_random(d->bias, 0, 1 / sqrt(input_dim));
  if(l->weight_initialization == SK_HE)
      tensor_fill_random(d->bias, 0, sqrt(2 / input_dim));

  l->output.dims[0] = 1;
  l->gradient.dims[0] = 1;
  l->data = d;
}
