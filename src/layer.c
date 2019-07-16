#include <string.h>

#include <conf.h>
#include <layer.h>

int sk_contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}

void sk_fully_connected_forward(Layer *l, Tensor *p, size_t t){
  
}

void sk_layer_forward(Layer *l, Tensor *p, size_t t){
  switch(l->layertype){
    case SK_FF:{

    }
    case SK_RC:{

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
    l->loutput        = create_tensor(SIEKNET_CPU, l->size);
  }else{
    l->output         = create_tensor(SIEKNET_CPU, 0, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, 0, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, 0, num_inputs);
  }
}

