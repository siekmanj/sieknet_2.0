#ifndef SIEKNET_LAYER_H
#define SIEKNET_LAYER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <sieknet.h>

Layer *layer_from_name(Network *, const char *);
int contains_layer(Layer **, Layer *, size_t);

#endif
