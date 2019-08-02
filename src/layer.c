#include <string.h>

#include <conf.h>
#include <layer.h>
#include <tensor.h>

#include <fc_layer.h>

int sk_contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}

void sk_layer_parse(Layer *l, char *identifier, char **remaining){
  switch(l->layertype){
    case SK_FF:
    case SK_RC:{
      sk_fc_layer_parse_attribute(l, identifier, remaining);
      break;
    }
    default:{
      SK_ERROR("Parse not implemented for this layer type.");
    }
  }
}

void sk_layer_allocate(Layer *l){
  switch(l->layertype){
    case SK_FF:
    case SK_RC:{
      sk_fc_layer_allocate(l);
      break;
    }
    case SK_LSTM:{
      SK_ERROR("LSTM not implemented.");
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
}

void sk_layer_initialize(Layer *l, Tensor p){
  switch(l->layertype){
    case SK_FF:
    case SK_RC:{
      sk_fc_layer_initialize(l, p);
      break;
    }
    case SK_LSTM:{
      SK_ERROR("LSTM not implemented.");
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
}

void (*sk_logistic_to_fn(SK_LOGISTIC l))(Tensor t){
  switch(l){
    case SK_SIGMOID:{
      return tensor_sigmoid;
      break;
    }
    case SK_TANH:{
      return tensor_tanh;
      break;
    }
    case SK_RELU:{
      return tensor_relu;
      break;
    }
    case SK_SOFTMAX:{
      return tensor_softmax;
      break;
    }
    default:{
      SK_ERROR("Logistic function not implemented.");
      break;
    }
  }
}
