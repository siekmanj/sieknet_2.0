#include <stdio.h>
#include <stdlib.h>

#include <util.h>
#include <conf.h>
#include <tensor.h>
#include <layer.h>

typedef struct fc_data_{
  Tensor bias;
  Tensor *weights;
} FC_layer_data;

void sk_fc_layer_forward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor b = d->bias;
  Tensor y = l->output.n == 1 ? l->output : get_subtensor(l->output, t);

  /* Zero the output tensor for this timestep */
  tensor_zero(y);

  /* Begin by elementwise-adding the bias to the output of this layer */
  tensor_elementwise_add(b, y, y);

  /* Loop through all the input layers and do a matrix mult */
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    Tensor w = d->weights[i];

    /* Get the subtensor for this timestep */
    Tensor x = in->rank >= l->rank ? in->loutput : in->output;
    x        = x.n == 1 ? x : get_subtensor(x, t);

    /* Matrix multiplication between weights and input */
    tensor_mmult(w, x, y);
  }
  //l->nonlinearity(y);
  tensor_print(y);
}

void sk_fc_layer_backward(Layer *l, const Tensor p, size_t t){
  SK_ERROR("Not implemented!");
}

void sk_fc_layer_parse(Layer *l, const char *identifier, char *remaining){

}

void sk_fc_layer_allocate(Layer *l, int recurrent){
  size_t num_inputs = l->num_params = 0;
  size_t params_per_neuron = 1;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += params_per_neuron * (l->size * in->size) + l->size;
    num_inputs += in->size;
  }

  if(recurrent){
    l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, num_inputs);
  }else{
    l->output         = create_tensor(SIEKNET_CPU, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, num_inputs);
  }
  l->loutput  = create_tensor(SIEKNET_CPU, l->size);
  l->forward  = sk_fc_layer_forward;
  l->backward = sk_fc_layer_backward;
  l->data     = (void *)calloc(1, sizeof(FC_layer_data));
  FC_layer_data *d = (FC_layer_data *)l->data;
  d->weights = calloc(l->num_input_layers, sizeof(Tensor));
}

void sk_fc_layer_initialize(Layer *l, Tensor p){

  /* 
   * Perform weight initialization according to the desired scheme. 
   * Default is Xavier initialization.
   */
  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;
  switch(l->weight_initialization){
    case SK_XAVIER:{
      float *theta = &((float*)p.data)[p.data_offset + l->param_idx];
      for(int i = 0; i < l->num_params; i++){
        theta[i] = normal(0, 1 / sqrt(input_dim));
      }
      break;
    }
    case SK_HE:{
      float *theta = &((float*)p.data)[p.data_offset + l->param_idx];
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

  d->bias             = create_tensor(SIEKNET_CPU, l->size);
  free(d->bias.data);
  d->bias.data        = p.data;
  d->bias.data_offset = param_offset;
  param_offset        += l->size;

  for(int i = 0; i < l->num_input_layers; i++){
    d->weights[i]             = create_tensor(SIEKNET_CPU, l->size, l->input_layers[i]->size);
    free(d->weights[i].data);
    d->weights[i].data        = p.data;
    d->weights[i].data_offset = param_offset;
    param_offset              += l->size * l->input_layers[i]->size;
  }
  
}

