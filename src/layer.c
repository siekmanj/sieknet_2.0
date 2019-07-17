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

void sk_fully_connected_forward(Layer *l, const Tensor p, size_t t){
  size_t param_idx = l->param_idx;
  for(int j = 0; j < l->size; j++){
    param_idx++;

    Tensor y = l->output;
    y.data_offset = get_flat_idx(y, &t, 1);

    for(int i = 0; i < l->num_input_layers; i++){
      Layer *in = l->input_layers[i];

      Tensor w = p;
      w.data_offset = param_idx;
      w.dims        = &in->size;
      w.n           = 1;

      Tensor x;
      if(in->rank >= l->rank)
        x = in->loutput;
      else
        x = in->output;

      x.data_offset = get_flat_idx(x, &t, 1);


      Tensor *y = l->output;

      tensor_reduce_dot(&w, 0, 0, x, 0, 0, y, j, in->size);


    }
    /*
      size_t idx_x = get_flat_idx(x, &t, 1);
      size_t idx_w = l->param_idx + 1;
      size_t idx_y = j;

      tensor_inner_product(x, idx_x, 0,
                           w, idx_w, 0,
                           y, idx_y, in->size);

    */
  }
  for(int i = 0; i < l->size; i++){
    //Tensor *b, *y;
    //tensor_add(
  }
}

void sk_layer_forward(Layer *l, const Tensor p, size_t t){
  switch(l->layertype){
    case SK_FF: // Intentional fall-through
    case SK_RC:{
      sk_fully_connected_forward(l, p, t);
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
    l->output         = create_tensor(SIEKNET_CPU, 0, l->size);
    l->gradient       = create_tensor(SIEKNET_CPU, 0, l->size);
    l->input_gradient = create_tensor(SIEKNET_CPU, 0, num_inputs);
  }
  l->loutput = create_tensor(SIEKNET_CPU, l->size);
}

