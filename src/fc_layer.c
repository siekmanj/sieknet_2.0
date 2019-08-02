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
  l->nonlinearity(y);
}

void sk_fc_layer_backward(Layer *l, size_t t){
  SK_ERROR("Not implemented!");
}

static const char *sk_fc_layer_identifiers[]   = {"logistic", "size", "type", "input", "name"};
void sk_fc_layer_parse_attribute(Layer *l, const char *identifier, char **remaining){
  SK_ERROR("This needs to be implemented.");
#if 0
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  int offset;

  if(sscanf(*remaining, "%s%n", buff, &offset) != EOF){

    char *original = *remaining;
    *remaining = *remaining + offset;

    if(!strcmp(identifier, "input")){
      if(l->input_names)
        SK_ERROR("a layer can only have one input layer(s) field.");

      size_t num_layers = 0;

      while(1){
        // Count the number of input layers
        if(get_token_type(buff) == VALUE){
          num_layers++;

          if(!(sscanf(*remaining, "%s%n", buff, &offset) != EOF))
            SK_ERROR("unexpected EOF while reading file.");
          *remaining += offset;
        }
        else
          break;
      }
      *remaining = original;
      l->input_names = calloc(num_layers, sizeof(char*));
      l->num_input_layers = num_layers;
      for(int i = 0; i < num_layers; i++){
        if((sscanf(*remaining, "%s%n", buff, &offset) != EOF)){
          *remaining += offset;

          char *name = calloc(strlen(buff), sizeof(char));
          strcpy(name, buff);
          l->input_names[i] = name;
        }
        else
          SK_ERROR("unexpected end of field while reading layer input names");
      }
    }
    else if(!strcmp(identifier, "logistic")){
      if(l->logistic != -1)
        SK_ERROR("a layer can have only one logistic function field, but found logistic functions '%d' (%s) and '%s'.", l->logistic, sk_logistics[l->logistic], buff);

      int logistic_idx = contains(buff, sk_logistics, STATIC_LEN(sk_logistics)); 
      if(logistic_idx != -1)
        l->logistic = logistic_idx;
      else
        SK_ERROR("could not find logistic function '%s'", buff);
    }
    else if(!strcmp(identifier, "type")){
      if(l->layertype != -1)
        SK_ERROR("a layer can only have one type field, but found types '%d' (%s) and '%s'.", l->layertype, sk_layertypes[l->layertype], buff);

      int layertype_idx = contains(buff, sk_layertypes, STATIC_LEN(sk_layertypes));
      if(layertype_idx != -1){
        l->layertype = layertype_idx;
      }else
        SK_ERROR("could not find layer type '%s'", buff);
    }
    else if(!strcmp(identifier, "size")){
      if(l->size)
        SK_ERROR("a layer can have only one size field, but found sizes '%lu' and '%s'.", l->size, buff);

      int size;
      if(sscanf(buff, "%d%n", &size, &offset)){
        l->size = size;
        if(l->size <= 0)
          SK_ERROR("layer size must be strictly greater than zero (got %lu).", l->size);
      }else{
        SK_ERROR("unable to parse layer size from '%s'", buff);
      }
    }
    else if(!strcmp(identifier, "name")){
      if(l->name)
        SK_ERROR("a layer can only have one name field, but found names '%s' and '%s'", l->name, buff);

      char *name = calloc(strlen(buff), sizeof(char));
      strcpy(name, buff);
      l->name = name;
    }
    else
      SK_ERROR("unrecognized identifier");
  }else
    SK_ERROR("unexpected EOF while reading file.");
#endif
}

void sk_fc_layer_allocate(Layer *l, int recurrent){
  size_t num_inputs = l->num_params = 0;
  size_t params_per_neuron = 1;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += params_per_neuron * (l->size * in->size) + l->size;
    num_inputs += in->size;
  }

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->input_gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, num_inputs);

  l->loutput      = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_fc_layer_forward;
  l->backward     = sk_fc_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(l->logistic);

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

