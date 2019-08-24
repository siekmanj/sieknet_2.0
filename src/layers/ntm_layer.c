#include <stdio.h>
#include <stdlib.h>

#include <conf.h>
#include <tensor.h>
#include <layer.h>
#include <parser.h>

/*
 * k - key vector
 * b - key strength (scalar)
 * g - blending factor (scalar)
 * s - shift weighting (random scalar?)
 * y - sharpening exponent (scalar)
 * e - erase vector
 * a - add vector
 */
#ifdef SIEKNET_NTM

typedef enum head_mode_{NTM_READ, NTM_WRITE} NTM_HEAD_MODE;

typedef struct ntm_head_{
  
  Tensor *key_weights;
  Tensor key_bias;
  Tensor key;

  Tensor *key_strength_weights;
  Tensor key_strength_bias;
  Tensor key_strength;

  Tensor *interpolation_gate_weights;
  Tensor interpolation_gate_bias;
  Tensor interpolation_gate;

  Tensor *sharpening_factor_weights;
  Tensor sharpening_factor_bias;
  Tensor sharpening_factor;

  Tensor *shift_factor_weights;
  Tensor shift_factor_bias;
  Tensor shift_factor;



  NTM_HEAD_MODE mode;

} NTM_head;

typedef struct ntm_data_{

  Tensor memory;
  Tensor logits;
  Tensor jacobian;

} NTM_layer_data;

void sk_ntm_layer_forward(Layer *l, size_t t){
  NTM_layer_data *d = (NTM_layer_data *)l->data;

  Tensor beta = ...;
  Tensor key = ...;
  Tensor mem = get_subtensor(d->memory, t);

  Tensor logits = get_subtensor(d->logits, t);

  tensor_cosine_similarity(key, mem, logits);
  tensor_scalar_mul(logits, beta);

  tensor_softmax(logits, jacobian);
}

void sk_ntm_layer_backward(Layer *l, size_t t){}

void sk_ntm_layer_wipe(Layer *l){}

void sk_ntm_layer_parse(Layer *l, char *src){}

void sk_ntm_layer_allocate(Layer *l){}

void sk_ntm_layer_initialize(Layer *l, Tensor p, Tensor g){}
#endif

