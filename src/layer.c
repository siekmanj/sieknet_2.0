#include <string.h>

#include <conf.h>
#include <layer.h>
#include <tensor.h>

int sk_contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}


/* 
 * For a layer of size n,
 * l->param_idx
 *     |
 *     V
 * ... b1 b2 ... bn n0_w0_l0 n1_w1_l0 ... w_m_l0 w_0_l1 w_1_l1 ... w_k_l1
 */

void sk_fc_forward(Layer *l, const Tensor p, size_t t){
  size_t param_idx = l->param_idx;

  Tensor b = p;
  b.data_offset = param_idx;
  b.dims        = &l->size;
  b.n           = 1;

  Tensor y = get_subtensor(l->output, t);

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

    tensor_mmult(w, 1, 0,
                 x, 0, 0,
                 y, 0, 0);

    
  }
}

void sk_layer_forward(Layer *l, const Tensor p, size_t t){
  switch(l->layertype){
    case SK_FF:
    case SK_RC:{
      sk_fc_forward(l, p, t);
      break;
    }
    case SK_LSTM:{
      SK_ERROR("LSTM forward pass not implemented.");
      break;
    }
    case SK_GRU:{
      SK_ERROR("GRU forward pass not implemented.");
      break;
    }
    case SK_ATT:{
      SK_ERROR("Attention forward pass not implemented.");
      break;
    }
  }
}

void sk_initialize_layer(Layer *l, int recurrent){
  size_t num_inputs;
  num_inputs = l->num_params = 0;

  size_t params_per_neuron = 0;

  switch(l->layertype){
    case SK_FF:
    case SK_RC:{
      params_per_neuron = 1;
      break;
    }
    case SK_LSTM:{
      params_per_neuron = 4;
      break;
    }
    case SK_GRU:{
      SK_ERROR("GRU not implemented.");
      break;
    }
    case SK_ATT:{
      SK_ERROR("Attention not implemented.");
      break;
    }
  }

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += params_per_neuron * (l->size + 1) * in->size;

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
  l->loutput = create_tensor(SIEKNET_CPU, l->size);
}

