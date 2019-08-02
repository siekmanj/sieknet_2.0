#ifndef SIEKNET_FC_LAYER_H
#define SIEKNET_FC_LAYER_H

#include <layer.h>

void sk_fc_layer_allocate(Layer *);

void sk_fc_layer_forward(Layer *, size_t);

void sk_fc_layer_backward(Layer *, const Tensor, size_t);

void sk_fc_layer_initialize(Layer *, Tensor);

void sk_fc_layer_parse_attribute(Layer *, const char *, char **);

#endif
