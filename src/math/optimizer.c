#include <optimizer.h>

void sgd_step(Optimizer o){
  tensor_scalar_mul(o.gradient, o.lr);
  tensor_elementwise_sub(o.params, o.gradient, o.params);
  tensor_fill(o.gradient, 0.0f);
}

Optimizer create_optimizer(Tensor params, Tensor gradient, SK_OPTIMIZER_TYPE t){
  Optimizer o = {0};

  o.params = params;
  o.gradient = gradient;

  o.step = sgd_step;

  o.lr = 1e-5;
  o.momentum = 0;

  if(t < 0)
    SK_ERROR("invalid type.");
  else if(t == SK_SGD);
  else if(t == SK_MOMENTUM);
  return o;
}

