#include <ddpg.h>
#include <math.h>

void ddpg_update_policy(DDPG d){
  if(d.n < d.minibatch_size)
    return;

  /*
   * Sample from buffer
   */
  Transition minibatch[d.minibatch_size];
  for(int i = 0; i < d.minibatch_size; i++){
    int randidx = rand() % (d.n-1);
    minibatch[i] = d.replay_buffer[randidx];
  }

  Layer *critic = sk_layer_from_name(d.policy, "critic");
  if(!critic)
    SK_ERROR("Unable to find layer with name 'critic'");

  Layer *actor= sk_layer_from_name(d.policy, "actor");
  if(!actor)
    SK_ERROR("Unable to find layer with name 'actor'");

  /*
   * Compute the target Q values.
   */
  tensor_copy(d.target_policy, d.policy->params);

  d.policy->t = 0;
  Tensor target_q = d._q_buffer;
  target_q.dims[0] = d.minibatch_size;

  printf("BEFORE:\n");
  tensor_fill(target_q, 0.0f);
  tensor_print(target_q);
  for(int i = 0; i < d.minibatch_size; i++){
    if(!minibatch[i].terminal){
      sk_forward(d.policy, minibatch[i].next_state);
      float critic_q = tensor_at(critic->output, i);

      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward + d.discount * critic_q;
    }else
      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward;

    tensor_print(target_q);
    //tensor_print(critic->output);
  }

  printf("AFTER:\n");
  tensor_print(target_q);
  getchar();
  /*
   *
   */

  /*
   *
   */
}

void ddpg_append_transition(DDPG *d, Tensor state, Tensor action, Tensor next_state, float reward, int terminal){
  if(d->n < d->num_timesteps-1){
    tensor_copy(state, d->replay_buffer[d->n].state);
    tensor_copy(action, d->replay_buffer[d->n].action);
    tensor_copy(next_state, d->replay_buffer[d->n].next_state);
    d->replay_buffer[d->n].reward = reward;
    d->replay_buffer[d->n].terminal = terminal;
    d->n++;
  }else
    SK_ERROR("TODO: Handle this case.\n");
}

DDPG create_ddpg(Network *networks, size_t action_space, size_t state_space, size_t num_threads, size_t num_timesteps){
  DDPG d = {0};

  if(networks[0].is_recurrent)
    SK_ERROR("DDPG only works with ff policies.");

  d.policy = networks;
  d.policy_gradient = tensor_clone(SIEKNET_CPU, networks[0].param_grad);
  d.target_policy   = tensor_clone(SIEKNET_CPU, networks[0].params);
  d.current_policy  = tensor_clone(SIEKNET_CPU, networks[0].params);

  d._q_buffer = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 1);

  if(!networks)
    SK_ERROR("Received null ptr for networks.");

  d.replay_buffer = (Transition *)malloc(sizeof(Transition) * num_timesteps);
  for(int i = 0; i < num_timesteps; i++){
    d.replay_buffer[i].state      = create_tensor(SIEKNET_CPU, state_space);
    d.replay_buffer[i].next_state = create_tensor(SIEKNET_CPU, state_space);
    d.replay_buffer[i].action     = create_tensor(SIEKNET_CPU, action_space);
    d.replay_buffer[i].reward     = -INFINITY;
    d.replay_buffer[i].terminal   = 0;
  }
  
  d.n = 0;
  d.minibatch_size = 4;
  d.num_timesteps = num_timesteps;
  d.discount = 0.99;

  return  d;
}
