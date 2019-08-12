#include <stdio.h>
#include <stdlib.h>

#include <util.h>
#include <conf.h>
#include <tensor.h>
#include <layer.h>

typedef struct fc_data_{
  Tensor bias;
  Tensor *weights;
  Tensor bias_grad;
  Tensor *weight_grad;
  Tensor intermediate_grad;
} FC_layer_data;

void sk_fc_layer_forward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor b = d->bias;
  Tensor y = get_subtensor(l->output, t);
  Tensor dy = get_subtensor(d->intermediate_grad, t);

  d->intermediate_grad.dims[0] = t + 1;

  /* Zero the output tensor for this timestep */
  tensor_zero(y);

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
  tensor_elementwise_add(b, y, y);
  l->nonlinearity(y, dy);
}

void sk_fc_layer_backward(Layer *l, size_t t){
  FC_layer_data *d = (FC_layer_data*)l->data;

  Tensor o = get_subtensor(l->gradient, t);
  Tensor g = get_subtensor(d->intermediate_grad, t);
  tensor_elementwise_mul(o, g, g); 

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    Tensor w  = d->weights[i];
    Tensor dw = d->weight_grad[i];

    Tensor x  = {0};
    Tensor dx = {0};
    if(in->rank >= l->rank){ // If this is a recurrent connection
      if(t > 0){
        x = get_subtensor(in->output, t-1);
        if(in->gradient.data)
          dx = get_subtensor(in->gradient, t-1);
      }else
        continue;
    }else{
      x = get_subtensor(in->output, t);
      if(in->gradient.data)
        dx = get_subtensor(in->gradient, t);
    }

    /* Compute weight gradients */
    tensor_mmult(x, g, dw); // dW = x * g

    /* Compute input gradients if needed */
    if(dx.data)
      tensor_mmult(w, g, dx); // dX = g * w
  }
	tensor_zero(o);
  /* Compute bias gradients */
  Tensor db = d->bias_grad;
  tensor_elementwise_add(g, db, db);
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

void sk_fc_layer_allocate(Layer *l){
  l->num_params = 0;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += (l->size * in->size) + l->size;
  }

  l->output         = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  l->loutput      = create_tensor(SIEKNET_CPU, l->size);

  l->forward      = sk_fc_layer_forward;
  l->backward     = sk_fc_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(l->logistic);

  FC_layer_data *d     = calloc(1, sizeof(FC_layer_data));
  d->weights           = calloc(l->num_input_layers, sizeof(Tensor));
  d->weight_grad       = calloc(l->num_input_layers, sizeof(Tensor));
  d->intermediate_grad = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->data = d;
}

void sk_fc_layer_initialize(Layer *l, Tensor p, Tensor g){
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

  d->bias      = get_subtensor_reshape(p, param_offset, l->size);
  d->bias_grad = get_subtensor_reshape(g, param_offset, l->size);

  param_offset += l->size;

  for(int i = 0; i < l->num_input_layers; i++){

    d->weights[i]     = get_subtensor_reshape(p, param_offset, l->input_layers[i]->size, l->size);
    d->weight_grad[i] = get_subtensor_reshape(g, param_offset, l->input_layers[i]->size, l->size);

    param_offset += l->size * l->input_layers[i]->size;
  }

  l->output.dims[0] = 1;
  l->gradient.dims[0] = 1;

}

