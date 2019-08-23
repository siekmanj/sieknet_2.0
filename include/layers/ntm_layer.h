#ifndef SIEKNET_NTM_LAYER_H
#define SIEKNET_NTM_LAYER_H

#include <layer.h>
#include <tensor.h>

static const char sk_ntm_layer_identifier[] = "[ntm layer]";

void sk_ntm_layer_forward(Layer *, size_t)

void sk_ntm_layer_backward(Layer *, size_t);

void sk_ntm_layer_wipe(Layer *);

void sk_ntm_layer_parse(Layer *, char *);

void sk_ntm_layer_allocate(Layer *);

void sk_ntm_layer_initialize(Layer *, Tensor, Tensor);

#endif
