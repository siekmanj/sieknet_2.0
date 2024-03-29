#ifndef SIEKNET_FC_LAYER_H
#define SIEKNET_FC_LAYER_H

#include <tensor.h>

static const char sk_fc_layer_identifier[] = "[fully_connected layer]";

void sk_fc_layer_parse(Layer *, const char *);

void sk_fc_layer_count_params(Layer *);

void sk_fc_layer_initialize(Layer *, Tensor, Tensor);

void sk_fc_layer_forward(Layer *, size_t);

void sk_fc_layer_backward(Layer *, size_t);

void sk_fc_layer_wipe(Layer *);

void sk_fc_layer_dealloc(Layer *);

#endif
