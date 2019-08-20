#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <layer.h>
#include <parser.h>

typedef struct lstm_data_{
  Tensor bias;
  Tensor *weights;

  Tensor bias_grad;
  Tensor *weight_grad;

  Tensor cell_state;
  Tensor cell_state_tanh;
  Tensor last_cell_state;
  Tensor cell_grad;

  Tensor gates;
  Tensor gate_grads;

  Tensor cell_future_grad;
} LSTM_layer_data;

void sk_lstm_layer_forward(Layer *l, size_t t){
  LSTM_layer_data *d = (LSTM_layer_data*)l->data;

  Tensor input_nonl_y = get_subtensor(d->gates, t, 0);
  Tensor input_gate_y = get_subtensor(d->gates, t, 1);
  Tensor forgt_gate_y = get_subtensor(d->gates, t, 2);
  Tensor outpt_gate_y = get_subtensor(d->gates, t, 3);

  Tensor input_nonl_dy = get_subtensor(d->gate_grads, t, 0);
  Tensor input_gate_dy = get_subtensor(d->gate_grads, t, 1);
  Tensor forgt_gate_dy = get_subtensor(d->gate_grads, t, 2);
  Tensor outpt_gate_dy = get_subtensor(d->gate_grads, t, 3);

  d->gates.dims[0]            = t + 1;
  d->gate_grads.dims[0]       = t + 1;
  d->cell_grad.dims[0]        = t + 1;
  d->cell_state.dims[0]       = t + 1;
  d->cell_state_tanh.dims[9]  = t + 1;
  d->cell_future_grad.dims[0] = t + 1;

  tensor_fill(get_subtensor(d->gates, t), 0.0f);

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    Tensor w = d->weights[i];

    /* Get the subtensor for this timestep */
    Tensor x = in->rank >= l->rank ? in->loutput : in->output;
    x        = x.n == 1 ? x : get_subtensor(x, t);

    Tensor input_nonl_w = get_subtensor(w, 0);
    Tensor input_gate_w = get_subtensor(w, 1);
    Tensor forgt_gate_w = get_subtensor(w, 2);
    Tensor outpt_gate_w = get_subtensor(w, 3);

    /* Do a weight transpose for all weights */
    tensor_transpose(d->weights[i], 0, 1);

    tensor_mmult(input_nonl_w, x, input_nonl_y);
    tensor_mmult(input_gate_w, x, input_gate_y);
    tensor_mmult(forgt_gate_w, x, forgt_gate_y);
    tensor_mmult(outpt_gate_w, x, outpt_gate_y);

    /* Restore the weight dimensions to their original shape */
    tensor_transpose(d->weights[i], 0, 1);
  }

  Tensor gates = get_subtensor(d->gates, t);
  tensor_elementwise_add(gates, d->bias, gates);

  /* Calculate all gate logistics and input nonlinearity */
  tensor_tanh_precompute(input_nonl_y, input_nonl_dy);
  tensor_sigmoid_precompute(input_gate_y, input_gate_dy);
  tensor_sigmoid_precompute(forgt_gate_y, forgt_gate_dy);
  tensor_sigmoid_precompute(outpt_gate_y, outpt_gate_dy);

  Tensor cell_state = get_subtensor(d->cell_state, t);
  Tensor cell_grad  = get_subtensor(d->cell_grad, t);

  Tensor cell_state_tanh = get_subtensor(d->cell_state_tanh, t);

  /* Multiply input nonlinearity and input gate together */
  tensor_elementwise_mul(input_nonl_y, input_gate_y, cell_state);

  /* Multiply previous cell state and forget gate together */
  tensor_elementwise_mul(d->last_cell_state, forgt_gate_y, d->last_cell_state);

  /* Add the previous two intermediate results */
  tensor_elementwise_add(d->last_cell_state, cell_state, cell_state);

  /* Pass the cell state through the tanh nonlinearity */
  tensor_copy(cell_state, cell_state_tanh);
  tensor_tanh_precompute(cell_state_tanh, cell_grad);

  Tensor y = get_subtensor(l->output, t);
  tensor_elementwise_mul(cell_state_tanh, outpt_gate_y, y);

  tensor_copy(cell_state, d->last_cell_state);
}

void sk_lstm_layer_backward(Layer *l, size_t t){
  LSTM_layer_data *d = (LSTM_layer_data*)l->data;

  const size_t max_t = l->gradient.dims[0] - 1;

  Tensor gradient = get_subtensor(l->gradient, t);

  Tensor input_nonl_y = get_subtensor(d->gates, t, 0);
  Tensor input_gate_y = get_subtensor(d->gates, t, 1);
  Tensor forgt_gate_y = get_subtensor(d->gates, t, 2);
  Tensor outpt_gate_y = get_subtensor(d->gates, t, 3);

  Tensor input_nonl_dy = get_subtensor(d->gate_grads, t, 0);
  Tensor input_gate_dy = get_subtensor(d->gate_grads, t, 1);
  Tensor forgt_gate_dy = get_subtensor(d->gate_grads, t, 2);
  Tensor outpt_gate_dy = get_subtensor(d->gate_grads, t, 3);

  Tensor cell_grad   = get_subtensor(d->cell_grad, t);
  Tensor cell_state_tanh = get_subtensor(d->cell_state_tanh, t);


  /* Derivative for cell state */
  tensor_elementwise_mul(cell_grad, outpt_gate_y, cell_grad);
  tensor_elementwise_mul(cell_grad, gradient, cell_grad);

  /* Add future gradient if one exists */
  if(t < max_t){
    Tensor cell_future = get_subtensor(d->cell_future_grad, t);
    tensor_elementwise_add(cell_grad, cell_future, cell_grad);
  }

  /* Derivative for output gate */
  tensor_elementwise_mul(outpt_gate_dy, cell_state_tanh, outpt_gate_dy);
  tensor_elementwise_mul(outpt_gate_dy, gradient, outpt_gate_dy);

  /* Derivative for input gate */
  tensor_elementwise_mul(input_gate_dy, input_nonl_y, input_gate_dy);
  tensor_elementwise_mul(input_gate_dy, cell_grad, input_gate_dy);

  /* Derivative for input nonlinearity */
  tensor_elementwise_mul(input_nonl_dy, input_gate_y, input_nonl_dy);
  tensor_elementwise_mul(input_nonl_dy, cell_grad, input_nonl_dy);

  if(t > 0){
    Tensor last_cell_state = get_subtensor(d->cell_state, t-1);
    Tensor last_future     = get_subtensor(d->cell_future_grad, t-1);

    /* Derivative for forget gate */
    tensor_elementwise_mul(forgt_gate_dy, cell_grad, forgt_gate_dy);
    tensor_elementwise_mul(forgt_gate_dy, last_cell_state, forgt_gate_dy);

    /* Derivative for last cell state */
    tensor_elementwise_mul(cell_grad, forgt_gate_y, last_future);
  }else
    tensor_fill(forgt_gate_dy, 0.0f);
  

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    
    int target_t = in->rank >= l->rank ? t - 1 : t;

    Tensor x = {0};
    Tensor dx = {0};

    if(target_t >= 0){
      x = get_subtensor(in->output, target_t);

      if(in->gradient.data)
        dx = get_subtensor(in->gradient, target_t);
      
    }else continue;

    for(int gate = 0; gate < 4; gate++){
      Tensor g = get_subtensor(d->gate_grads, t, gate);
      tensor_mmult(x, g, get_subtensor(d->weight_grad[i], gate));
      if(dx.data)
        tensor_mmult(get_subtensor(d->weights[i], gate), g, dx);
    }
  }
  tensor_elementwise_add(d->bias_grad, get_subtensor(d->gate_grads, t), d->bias_grad);
}

void sk_lstm_layer_wipe(Layer *l){
  LSTM_layer_data *d = (LSTM_layer_data*)l->data;
  tensor_fill(d->last_cell_state, 0.0f);
}

void sk_lstm_layer_parse(Layer *l, char *src){
  
  char *name;
  if(!sk_parser_find_string("name", src, &name))
    SK_ERROR("Unable to parse fc-layer attribute 'name'.");

  int size;
  if(!sk_parser_find_int("size", src, &size))
    SK_ERROR("Unable to parse fc-layer attribute 'size' for layer '%s'.\n", name);

  size_t num_names = 0;
  char **input_names;
  sk_parser_find_strings("input", src, &input_names, &num_names);

  SK_LOGISTIC logistic;
  char *logistic_src = NULL;
  if(sk_parser_find_string("logistic", src, &logistic_src))
    logistic = sk_layer_parse_logistic(logistic_src);
  else
    logistic = SK_TANH;
  if(logistic_src)
    free(logistic_src);

  l->size = size;
  l->input_names = input_names;
  l->logistic = logistic;
  l->num_input_layers = num_names;
  l->name = name;
}

void sk_lstm_layer_allocate(Layer *l){
  l->num_params = 0;

  for(int i = 0; i < l->num_input_layers; i++){
    Layer *in = l->input_layers[i];
    l->num_params += (4 * l->size * in->size) + l->size;
  }
  
  l->output   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->gradient = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  l->loutput  = create_tensor(SIEKNET_CPU, l->size);

  l->forward = sk_lstm_layer_forward;
  l->backward = sk_lstm_layer_backward;
  l->nonlinearity = sk_logistic_to_fn(l->logistic);
  l->wipe = sk_lstm_layer_wipe;

  LSTM_layer_data *d   = malloc(sizeof(LSTM_layer_data));
  d->weights           = calloc(l->num_input_layers, sizeof(Tensor));
  d->weight_grad       = calloc(l->num_input_layers, sizeof(Tensor));

  d->gates       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 4, l->size);
  d->gate_grads  = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 4, l->size);

  d->cell_state       = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  d->cell_state_tanh  = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  d->cell_grad        = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);
  d->cell_future_grad = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, l->size);

  d->last_cell_state = create_tensor(SIEKNET_CPU, l->size);
  
  l->data = d;
}

void sk_lstm_layer_initialize(Layer *l, Tensor p, Tensor g){
  size_t input_dim = 0;
  for(int i = 0; i < l->num_input_layers; i++)
    input_dim += l->input_layers[i]->size;

  /*
   * Set up weights and biases of this layer. We will use an internal struct
   * which is not used anywhere outside of this file (FC_layer_data). This 
   * is used to manage the forward and backward passes for fully connected
   * layers.
   */
  size_t param_offset = l->param_idx;
  LSTM_layer_data *d = (LSTM_layer_data *)l->data;

  d->bias      = get_subtensor_reshape(p, param_offset, 4, l->size);
  d->bias_grad = get_subtensor_reshape(g, param_offset, 4, l->size);

  param_offset += d->bias.size;

  for(int i = 0; i < l->num_input_layers; i++){

    d->weights[i]     = get_subtensor_reshape(p, param_offset, 4, l->input_layers[i]->size, l->size);
    d->weight_grad[i] = get_subtensor_reshape(g, param_offset, 4, l->input_layers[i]->size, l->size);

    if(l->weight_initialization == SK_XAVIER)
        tensor_fill_random(d->weights[i], 0, 1 / sqrt(input_dim));
    if(l->weight_initialization == SK_HE)
        tensor_fill_random(d->weights[i], 0, sqrt(2 / input_dim));

    param_offset += d->weights[i].size;
  }

  l->output.dims[0] = 1;
  l->gradient.dims[0] = 1;
}

