#include <stdio.h>
#include <stdlib.h>

#include <layer.h>
#include <tensor.h>
#include <sieknet.h>
#include <parser.h>

static void initialize_network(Network *n){
  /* Use a dummy layer to send input to the network */
  Layer *data = (Layer *)malloc(sizeof(Layer));
  data->size       = n->input_dimension;
  data->rank       = -1;
  data->name       = "DATA_IN";
  data->params_per_input = 0;

  data->output     = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, n->input_dimension);

  /*
   * Take the current input layer and add our dummy layer
   * as an input
   */
  n->data_layer = data;
  Layer **new_input_layers = (Layer **)malloc((n->input_layer->num_input_layers + 1) * sizeof(Layer *));
  new_input_layers[0] = data;
  for(int i = 1; i < n->input_layer->num_input_layers + 1; i++)
    new_input_layers[i] = n->input_layer->input_layers[i-1];
  free(n->input_layer->input_layers);
  n->input_layer->input_layers = new_input_layers;
  n->input_layer->num_input_layers++;
  
  /* Allocate tensor memory and count the number of parameters in the network */
  size_t param_idx = 0;
  for(int i = 0; i < n->depth; i++){
    Layer *l = n->layers[i];
    l->param_idx  = param_idx;
    sk_layer_allocate(l);
    param_idx += l->num_params;
  }
  n->params     = create_tensor(SIEKNET_CPU, param_idx);
  n->param_grad = create_tensor(SIEKNET_CPU, param_idx);

  /* Initialize layer weights and variables */
  for(int i = 0; i < n->depth; i++)
    sk_layer_initialize(n->layers[i], n->params);

  n->t = 0;
  n->trainable = 1;
}

Layer *sk_layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    if(!strcmp(n->layers[i]->name, name))
      return n->layers[i];
  }
  return NULL;
}

Network sk_create_network(const char *skfile){
  Network n = {0};

  /* 
   * Retrieve layer names + sizes, input layer names, 
   * logistic functions, layer types, and network name.
   */
  parse_network(&n, skfile);

  /*
   * Construct directed graph, assign input & output layers,
   * determine order of execution and recurrent inputs.
   */
  build_network(&n);

  /*
   *  Allocate memory, initialize layer tensor objects
   */
  initialize_network(&n);

  return n;
}

static void sk_run_inference(Network *n){
  /* 
   * Run the forward pass for each individual layer.
   */
  for(int i = 0; i < n->depth; i++)
    n->layers[i]->forward(n->layers[i], n->t);

  /* 
   * The below is a hack and is not guaranteed to work for transposed tensors - fix tensor_copy asap
   */ 
  for(int i = 0; i < n->depth; i++){
    Tensor o = get_subtensor(n->layers[i]->output, n->t);
    tensor_copy(o, n->layers[i]->loutput);
  }
}

void sk_forward(Network *n, Tensor x){
  if(x.n != 2 && x.n != 1)
    SK_ERROR("Expected a tensor of dimension 1 or 2 as input, but got %lu dimensions.\n", x.n);

  size_t sequence_length = x.n == 2 ? x.dims[0] : 1;
  size_t input_dimension = x.n == 2 ? x.dims[1] : x.dims[0];

  if(sequence_length > SIEKNET_MAX_UNROLL_LENGTH)
    SK_ERROR("Cannot have a sequence (%lu) longer than the max unroll length (%d).", sequence_length, SIEKNET_MAX_UNROLL_LENGTH);

  if(input_dimension != n->input_dimension)
    SK_ERROR("Expected input dimension %lu but got %lu.", n->input_dimension, input_dimension);

  if(x.n == 2){
    Tensor tmp = n->data_layer->output;       // Temporarily store the info in the data layer elsewhere.
    n->data_layer->output = x;                // Replace the data layer tensor with the input tensor.
    for(int t = 0; t < sequence_length; t++){
      n->t = n->trainable ? n->t + 1 : 0;     // If the network is in trainable mode, increment t.
      sk_run_inference(n);                    // Run inference for this time step.
    }
    n->data_layer->output = tmp;              // Restore the data layer
  }
  else if(x.n == 1){
    /* 
     * The below is a hack and is not guaranteed to work with transposed tensors - fix tensor_copy asap
     */
    Tensor input = get_subtensor(n->data_layer->output, n->t);
    tensor_copy(x, input);

    n->t = n->trainable ? n->t + 1 : 0;          // If the network is in trainable mode, increment t.
    sk_run_inference(n);                         // Run inference for this time step.
  }
}

float sk_cost(Network *n, Layer *l, Tensor y, SK_COST_FN cost){
  if(y.n > 2)
    SK_ERROR("Cannot currently handle tensors with dimension > 2 as labels.");

  if(y.n == 1 && y.dims[0] != l->size)
    SK_ERROR("Layer output dimension is %lu but cost tensor received was length %lu\n", l->size, y.dims[0]);

  if(y.n == 2 && y.dims[1] != l->size)
    SK_ERROR("Layer output dimension is %lu but cost tensor received was length %lu\n", l->size, y.dims[1]);

  float (*cost_function)(Tensor, Tensor, Tensor) = NULL;
  switch(cost){
    case SK_QUADRATIC_COST:{
      cost_function = tensor_quadratic_cost;
      break;
    }
    case SK_CROSS_ENTROPY_COST:{
      cost_function = tensor_cross_entropy_cost;
      break;
    }
    default:{
      SK_ERROR("Invalid cost function.");
      return 0.0f;
      break;
    }
  }

  if(y.n == 1){
    Tensor grad   = get_subtensor(l->gradient, n->t);
    Tensor output = get_subtensor(l->output, n->t);
    return cost_function(output, y, grad);
  }
  if(y.n == 2){
    if(y.dims[0] != n->t)
      SK_ERROR("Network timesteps are %lu, but label tensor is a sequence of length %lu. Expected these to match. Aborting.", n->t, y.dims[0]);

    float sum_cost = 0;
    for(int t = 0; t < y.dims[0]; t++){
      Tensor grad   = get_subtensor(l->gradient, t);
      Tensor output = get_subtensor(l->output, t);
      Tensor label  = get_subtensor(y, t);
      sum_cost += cost_function(output, label, grad);
    }
    return sum_cost;
  }
  return -1;
}

void sk_backward(Network *n){

}


