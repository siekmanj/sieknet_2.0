#ifndef SIEKNET_LSTM_LAYER_H
#define SIEKNET_LSTM_LAYER_H

#include <tensor.h>

static const char sk_lstm_layer_identifier[] = "[lstm layer]";

void sk_lstm_layer_parse(Layer *, const char *);

void sk_lstm_layer_count_params(Layer *);

void sk_lstm_layer_initialize(Layer *, Tensor, Tensor);

void sk_lstm_layer_forward(Layer *, size_t);

void sk_lstm_layer_backward(Layer *, size_t);

void sk_lstm_layer_wipe(Layer *);

void sk_lstm_layer_dealloc(Layer *);

#endif
