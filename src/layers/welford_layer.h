#ifndef SIEKNET_WELFORD_LAYER_H
#define SIEKNET_WELFORD_LAYER_H

#include <tensor.h>

static const char sk_welford_layer_identifier[] = "[welford layer]";

void sk_welford_layer_parse(Layer *, const char *);

void sk_welford_layer_count_params(Layer *);

void sk_welford_layer_initialize(Layer *);

void sk_welford_layer_forward(Layer *, size_t);

void sk_welford_layer_backward(Layer *, size_t);

void sk_welford_layer_wipe(Layer *);

#endif


