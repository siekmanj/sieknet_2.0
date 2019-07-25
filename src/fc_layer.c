#include <stdio.h>
#include <stdlib.h>

#include <util.h>
#include <conf.h>
#include <tensor.h>
#include <layer.h>

/* 
 * For a layer of size n,
 * l->param_idx
 *     |
 *     V
 * ... b1 b2 ... bn, n0_w0_l0 n1_w1_l0 ... 
 */
void sk_fc_layer_forward(Layer *l, const Tensor p, size_t t){
  size_t param_idx = l->param_idx;

  Tensor b = p;
  b.data_offset = param_idx;
  b.dims        = &l->size;
  b.n           = 1;

  Tensor y;
  if(l->output.n > 1)
    y = get_subtensor(l->output, t);
  else
    y = l->output;

#if 0
  /* Begin by elementwise-adding the bias to the output of this layer */
  tensor_elementwise_add(b, 0,  // Operand 1, no offset, axis 0
                         y, 0,  // Operand 2, no offset, axis 0
                         y, 0); // Destination tensor

#endif

  param_idx += l->size;

  /* Loop through all the input layers and do a matrix mult */
  for(int i = 0; i < l->num_input_layers; i++){

    Layer *in = l->input_layers[i];

    /* Create a new tensor from the network's parameter tensor with the correct shape */
    size_t w_dims[] = {l->size, in->size};
    Tensor w = p;
    w.data_offset = param_idx;
    w.dims        = w_dims;
    w.n           = 2;

    /* Get the subtensor for this timestep */
    Tensor x;

    if(in->rank >= l->rank)
      x = in->loutput;

    else if(l->input_layers[i]->output.n == 1)
      x = in->output;

    else
      x = get_subtensor(in->output, t);

#if 1
    printf("W:\n");
    tensor_print(w);
    printf("X:\n");
    tensor_print(x);
    printf("Y:\n");
    tensor_print(y);
#endif
    tensor_mmult(w, x, y);
  }
}

void sk_fc_layer_backward(Layer *l, const Tensor p, size_t t){
  SK_ERROR("Not implemented!");
}

void sk_fc_layer_allocate(Layer *l, int recurrent){
  size_t num_inputs = l->num_params = 0;
  size_t params_per_neuron = 1;

  printf("allocating\n");
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    printf("%lu += %lu\n", l->num_params, params_per_neuron * (l->size + 1) * in->size);
    l->num_params += params_per_neuron * (l->size + 1) * in->size;

    num_inputs += in->size;
  }
  printf("done allocating\n");

  if(recurrent){
    l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, num_inputs);
  }else{
    l->output         = create_tensor(SIEKNET_CPU, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, num_inputs);
  }
  l->loutput = create_tensor(SIEKNET_CPU, l->size);
  l->forward = sk_fc_layer_forward;
  l->backward = sk_fc_layer_backward;
}

void sk_fc_layer_initialize(Layer *l, Tensor p){
  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;

  switch(l->weight_initialization){
    case SK_XAVIER:{
      float *theta = &((float*)p.data)[p.data_offset + l->param_idx];
      for(int i = 0; i < l->num_params; i++)
        theta[i] = normal(0, 1 / sqrt(input_dim));

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
}

