#include <stdio.h>
#include <tensor.h>
#include <sieknet.h>

typedef struct transition_{
  Tensor state;
  Tensor action;
  float reward;
  //struct transition_ *parent;
  int terminal;
} Transition;

typedef struct ddpg_{
  Transition *transitions;
  Network *policy;

  Tensor policy_gradient;
  Tensor target_policy;
  Tensor current_policy;

  size_t minibatch_size;
  size_t num_timesteps;
  size_t n;

  void (*sample)(struct ddpg_);
  void (*update_policy)(struct ddpg_);
} DDPG;

DDPG create_ddpg(Network *, size_t, size_t, size_t, size_t);

void ddpg_append_transition(DDPG *, Tensor, Tensor, float, int);
void ddpg_update_policy(DDPG);
