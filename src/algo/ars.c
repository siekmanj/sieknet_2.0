#include <string.h>
#include <conf.h>
#include <ars.h>
#include <math.h>

#ifdef SIEKNET_USE_OMP
#include <omp.h>
#endif

static int ars_comparator(const void *one, const void *two){
	Delta *a = (Delta*)one;
	Delta *b = (Delta*)two;

	float max_a = MAX(a->r_pos, a->r_neg);
	float max_b = MAX(b->r_pos, b->r_neg);

	if(max_a < max_b)
		return 1;
	if(max_a > max_b)
		return -1;
	else 
		return 0;
}

static void ars_step(ARS r){
#if 0
#ifdef _OPENMP
  omp_set_num_threads(r.num_threads);
  #pragma omp parallel for default(none) shared(r)
#endif
#endif
  /* Do rollouts */
  for(int i = 0; i < r.directions; i++){
#ifdef _OPENMP
    size_t thread = omp_get_thread_num();
#else
    size_t thread = 0;
#endif

    tensor_elementwise_add(r.params, r.deltas[i].d, r.deltas[i].p);
    r.deltas[i].r_pos = r.f(r.deltas[i].p, thread);

    tensor_elementwise_sub(r.params, r.deltas[i].d, r.deltas[i].p);
    r.deltas[i].r_neg = r.f(r.deltas[i].p, thread);
  }

  int b = MIN(r.top_b, r.directions);

  if(b < r.directions)
    qsort(r.deltas, r.directions, sizeof(Delta), ars_comparator);

  if(r.algo == BASIC){
    for(int i = 0; i < b; i++){
      float weight = -(r.step_size / b) * (r.deltas[i].r_pos - r.deltas[i].r_neg);
      tensor_scalar_mul(r.deltas[i].d, weight * r.step_size);
    }
  }else if(r.algo == AUGMENTED){
    /* Mean and standard deviation of reward calculation */
    float mean = 0;
    float std  = 0;

    for(int i = 0; i < b; i++)
      mean += r.deltas[i].r_pos + r.deltas[i].r_neg;
    mean /= 2 * b;

    for(int i = 0; i < b; i++){
      float x = r.deltas[i].r_pos;
      std += (x - mean) * (x - mean);
      x = r.deltas[i].r_neg;
      std += (x - mean) * (x - mean);
    }
    std = sqrt(std/(2 * b));

    float weight = -1 / (b * std);
    for(int i = 0; i < b; i++){
      float reward = (r.deltas[i].r_pos - r.deltas[i].r_neg) / r.std;
      tensor_scalar_mul(r.deltas[i].d, weight * reward * r.step_size);
    }
  }
  
  for(int i = 0; i < b; i++)
    tensor_elementwise_add(r.params, r.deltas[i].d, r.params);

  for(int i = 0; i < r.directions; i++){
    tensor_fill_random(r.deltas[i].d, 0, r.std);
  }
}

ARS create_ars(float (*R)(Tensor, size_t), Tensor seed, size_t num_deltas, size_t num_threads){
  ARS r = {0};
	r.std = 0.02;
	r.step_size = 0.02;
	r.directions = num_deltas;
	r.top_b = r.directions;
  r.num_threads = num_threads;
  r.params = seed;
  r.update = tensor_clone(SIEKNET_CPU, seed);
  tensor_fill(r.update, 0.0f);

  r.f = R;
  r.step = ars_step;
  r.algo = AUGMENTED;

	r.deltas = calloc(num_deltas, sizeof(Delta));
	for(int i = 0; i < num_deltas; i++){
		r.deltas[i].d = tensor_clone(SIEKNET_CPU, seed);
		r.deltas[i].p = tensor_clone(SIEKNET_CPU, seed);
    tensor_fill_random(r.deltas[i].d, 0, r.std);
    tensor_fill(r.deltas[i].p, 0.0f);
	}
  return r;
}
