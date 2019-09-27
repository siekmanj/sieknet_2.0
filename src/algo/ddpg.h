#include <stdio.h>

#include <optimizer.h>
#include <tensor.h>
#include <sieknet.h>

typedef struct transition_{
  Tensor state;
  Tensor next_state;
  Tensor action;
  float reward;
  int terminal;
} Transition;

typedef struct ddpg_{
  Transition *replay_buffer;
  Network *policy;

  Tensor policy_gradient;
  Tensor target_policy;
  Tensor current_policy;

  Tensor _q_buffer;

  float discount;
  float tau;
  float actor_lr;
  float critic_lr;

  size_t minibatch_size;
  size_t num_timesteps;
  size_t n;

  void (*sample)(struct ddpg_);
  void (*update_policy)(struct ddpg_);

  Optimizer optimizer;
} DDPG;

DDPG create_ddpg(Network *, size_t, size_t, size_t, size_t);

void ddpg_append_transition(DDPG *, Tensor, Tensor, Tensor, float, int);
float ddpg_update_policy(DDPG);
