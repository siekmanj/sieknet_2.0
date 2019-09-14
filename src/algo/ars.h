#ifndef RANDOM_SEARCH_H
#define RANDOM_SEARCH_H

#include <stdlib.h>
#include <stdio.h>
#include <tensor.h>

typedef enum search_type_t {BASIC, AUGMENTED} SEARCH_TYPE;

#if 1

typedef struct delta_{
	Tensor d;
  Tensor p;
	float r_pos;
	float r_neg;
} Delta;

typedef struct ars_{
	float std;
	float step_size;

	size_t directions;
	size_t top_b;
	size_t num_params;
  size_t num_threads;

	Tensor params;
	Tensor update;
	Delta *deltas;

  float (*f)(Tensor, size_t);
  void (*step)(struct ars_);

	SEARCH_TYPE algo;
} ARS;

ARS create_ars(float (*R)(Tensor, size_t), Tensor, size_t, size_t);

//void ars_step(ARS);
#endif

#endif
