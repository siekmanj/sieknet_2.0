#include <string.h>

#include <conf.h>
#include <layer.h>
#include <tensor.h>
#include <parser.h>

#include <fc_layer.h>
#include <lstm_layer.h>

int sk_contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}

SK_LAYER_TYPE sk_layer_parse_identifier(const char *line){
  if(!line)
    SK_ERROR("Cannot parse null pointer.");
  else if(!strcmp(line, sk_fc_layer_identifier))
    return SK_FF;
  if(!strcmp(line, sk_lstm_layer_identifier))
    return SK_LSTM;
  return -1;
}

SK_LOGISTIC sk_layer_parse_logistic(const char *line){
  if(!strcmp(line, "sigmoid"))
    return SK_SIGMOID;
  if(!strcmp(line, "tanh"))
    return SK_TANH;
  if(!strcmp(line, "relu"))
    return SK_RELU;
  if(!strcmp(line, "softmax"))
    return SK_SOFTMAX;
  return -1;
}

void sk_layer_parse(Layer *l, char *src){
  char first_line[1024];
  sk_parser_get_line(&src, first_line, NULL);
  l->layertype = sk_layer_parse_identifier(first_line);

  switch(l->layertype){
    case SK_FF:
      sk_fc_layer_parse(l, src);
      break;
    case SK_LSTM:
      sk_lstm_layer_parse(l, src);
      break;
    default:{
      SK_ERROR("Parse not implemented for this layer type: %d.", l->layertype);
    }
  }
}

void sk_layer_allocate(Layer *l){
  switch(l->layertype){
    case SK_FF:
      sk_fc_layer_allocate(l);
      break;
    case SK_LSTM:
      sk_lstm_layer_allocate(l);
      break;
    case SK_GRU:
      SK_ERROR("GRU not implemented.");
      break;
    case SK_ATT:
      SK_ERROR("Attention not implemented.");
      break;
    default:
      SK_ERROR("Not implemented.");
      break;
  }
}

void sk_layer_initialize(Layer *l, Tensor p, Tensor g){
  switch(l->layertype){
    case SK_FF:
      sk_fc_layer_initialize(l, p, g);
      break;
    case SK_LSTM:
      sk_lstm_layer_initialize(l, p, g);
      break;
    case SK_GRU:{
      SK_ERROR("GRU not implemented.");
      break;
    }
    case SK_ATT:{
      SK_ERROR("Attention not implemented.");
      break;
    }
    default:{
      SK_ERROR("Not implemented.");
      break;
    }
  }
}

void (*sk_logistic_to_fn(SK_LOGISTIC l))(Tensor, Tensor){
  switch(l){
    case SK_SIGMOID:{
      return tensor_sigmoid_precompute;
      break;
    }
    case SK_TANH:{
      return tensor_tanh_precompute;
      break;
    }
    case SK_RELU:{
      return tensor_relu_precompute;
      break;
    }
    case SK_SOFTMAX:{
      return tensor_softmax_precompute;
      break;
    }
    default:{
      SK_ERROR("Logistic function not implemented.");
      break;
    }
  }
}
