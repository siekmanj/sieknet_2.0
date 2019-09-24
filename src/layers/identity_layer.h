#ifndef SIEKNET_IDENTITY_LAYER_H
#define SIEKNET_IDENTITY_LAYER_H

#include <tensor.h>

static const char sk_identity_layer_identifier[] = "[identity layer]";

void sk_identity_layer_parse(Layer *, const char *);

void sk_identity_layer_count_params(Layer *);

void sk_identity_layer_initialize(Layer *);

void sk_identity_layer_forward(Layer *, size_t);

void sk_identity_layer_backward(Layer *, size_t);

void sk_identity_layer_wipe(Layer *);

#endif


