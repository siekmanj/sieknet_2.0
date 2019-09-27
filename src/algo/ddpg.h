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

  Tensor target_policy, current_policy;
  Tensor actor_params, critic_params;
  Tensor actor_param_grad, critic_param_grad;

  Layer *critic_layer, *actor_layer, *state_layer;

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

  Optimizer critic_optimizer, actor_optimizer;
} DDPG;

DDPG create_ddpg(Network *, size_t, size_t, size_t, size_t);

void ddpg_append_transition(DDPG *, Tensor, Tensor, Tensor, float, int);
float ddpg_update_policy(DDPG);
