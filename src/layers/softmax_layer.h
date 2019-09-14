#ifndef SIEKNET_SOFTMAX_LAYER_H
#define SIEKNET_SOFTMAX_LAYER_H

#include <tensor.h>

static const char sk_softmax_layer_identifier[] = "[softmax layer]";

void sk_softmax_layer_parse(Layer *, const char *);

void sk_softmax_layer_count_params(Layer *);

void sk_softmax_layer_initialize(Layer *);

void sk_softmax_layer_forward(Layer *, size_t);

void sk_softmax_layer_backward(Layer *, size_t);

void sk_softmax_layer_wipe(Layer *);

#endif

