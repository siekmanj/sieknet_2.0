#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <parser.h>

#include <fc_layer.h>
#include <softmax_layer.h>

/*
 * k - key vector
 * b - key strength (scalar)
 * g - blending factor (scalar)
 * s - shift weighting (random scalar?)
 * y - sharpening exponent (scalar)
 * e - erase vector
 * a - add vector
 */

typedef enum head_mode_{NTM_READ, NTM_WRITE} NTM_HEAD_MODE;

typedef struct ntm_head_{
  // fully-connected layers
  Layer fc_key;
  Layer fc_key_strength;
  Layer fc_interpolation_gate;
  Layer fc_sharpening_factor;
  Layer fc_shift_factor;
  Layer fc_erase_vector;
  Layer fc_add_vector;

  Layer key_softmax;
  Layer shift_softmax;

  size_t num_params;

  NTM_HEAD_MODE mode;

} NTM_head;

typedef struct ntm_data_{

  Tensor memory;
  NTM_head *read_head;
  NTM_head *write_head;

  size_t mem_len;

} NTM_layer_data;

static void sk_ntm_head_forward(NTM_head *h, size_t t){
  //printf("********DOING HEAD FORWARD MODE %d\n", h->mode);
  sk_fc_layer_forward(&h->fc_key, t);
  sk_fc_layer_forward(&h->fc_key_strength, t);
  sk_fc_layer_forward(&h->fc_interpolation_gate, t);
  sk_fc_layer_forward(&h->fc_sharpening_factor, t);
  sk_fc_layer_forward(&h->fc_shift_factor, t);
  sk_fc_layer_forward(&h->fc_erase_vector, t);
  sk_fc_layer_forward(&h->fc_add_vector, t);

  //printf("FROM ISNIDE NTM HEAD FWD %p\n", tensor_raw(h->fc_key->output));
  //tensor_print(h->fc_key->output);

  sk_softmax_layer_forward(&h->key_softmax, t);
  sk_softmax_layer_forward(&h->shift_softmax, t);

  //printf("shoul have been %p\n", tensor_raw(h->key_softmax->output));
  tensor_print(get_subtensor(h->fc_key.output, t));
  tensor_print(get_subtensor(h->key_softmax.output, t));

  /*
  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];

    Tensor x = in->rank >= l->rank ? in->loutput : in->output;
    x        = x.n == 1 ? x : get_subtensor(x, t);
    printf("\tGetting input from '%s'\n", l->input_layers[i]->name);
    tensor_print(x);
  }
  */

}

void sk_ntm_layer_forward(Layer *l, size_t t){
  NTM_layer_data *d = (NTM_layer_data *)l->data;

  sk_ntm_head_forward(d->read_head, t);
  sk_ntm_head_forward(d->write_head, t);

  /* Zero the output tensor for this timestep */
  Tensor y = get_subtensor(l->output, t);
  tensor_fill(y, 0.0f);

  getchar();

}

void sk_ntm_layer_backward(Layer *l, size_t t){}

void sk_ntm_layer_wipe(Layer *l){}

void sk_ntm_layer_parse(Layer *l, char *src){
  NTM_layer_data *d = calloc(sizeof(NTM_layer_data), 1);
  d->mem_len = 32;

  if(!sk_parser_find_string("name", src, &l->name))
    SK_ERROR("Unable to parse fc-layer attribute 'name'.");

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);
  l->input_names = input_names;
  l->num_input_layers = num_names;

  l->data = d;

  l->size = 16;
}

void sk_ntm_layer_count_params(Layer *l){
  NTM_layer_data *d = (NTM_layer_data *)l->data;

  /*
   * key vector, add vector, erase vector (3xdim)
   * key strength, blending factor, exponent (3x1)
   * shift weighting (3x1)
   */
  size_t head_output_dim = 3 * d->mem_len + 3 + 3;

  l->num_params = 0;
  for(int i = 0; i < l->num_input_layers; i++){
    l->num_params += l->input_layers[i]->size * head_output_dim;
    l->num_params += l->input_layers[i]->size * head_output_dim;
  }
  l->num_consts = 0;
}

static Layer init_fc_sublayer(Layer *parent, Layer **inputs, size_t num_inputs, size_t size, SK_LOGISTIC logistic, Tensor p, Tensor g, size_t param_offset){
  Layer ret = {0};
  ret.name                  = "ntm-internal";
  ret.size                  = size;
  ret.num_input_layers      = num_inputs;
  ret.input_layers          = inputs;
  ret.rank                  = parent->rank;
  ret.logistic              = logistic;
  ret.weight_initialization = SK_XAVIER;
  ret.param_idx             = param_offset;
  //ret.num_params            = sk_fc_layer_count_params(&ret);
  
  sk_fc_layer_initialize(&ret, p, g);
  return ret;
}

static Layer init_softmax_sublayer(Layer *input){
  Layer ret = {0};
  ret.input_layers          = (Layer**)malloc(sizeof(Layer*));
  ret.name                  = "ntm-internal";
  ret.size                  = input->size;
  ret.num_input_layers      = 1;
  *ret.input_layers         = input;
  ret.rank                  = input->rank + 1;
  sk_softmax_layer_initialize(&ret);
  return ret;
}

static NTM_head *create_ntm_head(Layer *l, size_t mem_len, NTM_HEAD_MODE mode, Tensor p, Tensor g, size_t param_idx){
  NTM_head *h = (NTM_head*)malloc(sizeof(NTM_head));
  size_t param_offset = param_idx;
  h->fc_key          = init_fc_sublayer(l, l->input_layers, l->num_input_layers, mem_len, SK_LINEAR, p, g, param_offset);
  param_offset += h->fc_key.num_params;

  h->fc_key_strength = init_fc_sublayer(l, l->input_layers, l->num_input_layers, 1, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_key_strength.num_params;

  h->fc_interpolation_gate = init_fc_sublayer(l, l->input_layers, l->num_input_layers, 1, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_interpolation_gate.num_params;

  h->fc_sharpening_factor = init_fc_sublayer(l, l->input_layers, l->num_input_layers, 1, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_sharpening_factor.num_params;

  h->fc_shift_factor= init_fc_sublayer(l, l->input_layers, l->num_input_layers, 3, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_shift_factor.num_params;

  h->fc_erase_vector = init_fc_sublayer(l, l->input_layers, l->num_input_layers, 1, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_erase_vector.num_params;

  h->fc_add_vector = init_fc_sublayer(l, l->input_layers, l->num_input_layers, 1, SK_SIGMOID, p, g, param_offset);
  param_offset += h->fc_add_vector.num_params;

  h->num_params = param_offset - param_idx;

  h->key_softmax = init_softmax_sublayer(&h->fc_key);

  h->shift_softmax = init_softmax_sublayer(&h->fc_shift_factor);

  h->mode = mode;
  /*
   * ...
   */
  return h;
}

void sk_ntm_layer_initialize(Layer *l, Tensor p, Tensor g){
  NTM_layer_data *d = (NTM_layer_data *)l->data;
  d->memory = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 32, 32);

  size_t param_offset = l->param_idx;
  d->read_head  = create_ntm_head(l, d->memory.dims[d->memory.n - 1], NTM_READ, p, g, param_offset);
  param_offset += d->read_head->num_params;
  d->write_head = create_ntm_head(l, d->memory.dims[d->memory.n - 1], NTM_WRITE, p, g, param_offset);

  l->output   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->loutput  = create_tensor(SIEKNET_CPU, l->size);
  
  l->forward = sk_ntm_layer_forward;
  l->backward = sk_ntm_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(SK_LINEAR);
  l->wipe = sk_ntm_layer_wipe;
  
  l->data = d;
}
