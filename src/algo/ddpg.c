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

  Layer *actor = sk_layer_from_name(d.policy, "actor");
  if(!actor)
    SK_ERROR("Unable to find layer with name 'actor'");

  Layer *state = sk_layer_from_name(d.policy, "state");
  if(!state)
    SK_ERROR("Unable to find layer with name 'state'");

  /*
   * Compute the target Q values.
   */
  d.policy->t = 0;

  Tensor target_q = d._q_buffer;
  target_q.dims[0] = d.minibatch_size;
  tensor_fill(target_q, 0.0f);

  tensor_copy(d.target_policy, d.policy->params); // Copy target parameters to policy
  for(int i = 0; i < d.minibatch_size; i++){
    if(!minibatch[i].terminal){
      sk_forward(d.policy, minibatch[i].next_state);
      float critic_q = tensor_at(critic->output, i); // Get the critic value

      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward + d.discount * critic_q;
    }else
      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward;
  }

  /*
   * Compute the current Q values.
   */
  d.policy->t = 0;
  tensor_copy(d.current_policy, d.policy->params); // Copy current parameters to policy
  for(int i = 0; i < d.minibatch_size; i++){
    tensor_copy(minibatch[i].action, get_subtensor(actor->output, i));
    tensor_copy(minibatch[i].state, get_subtensor(state->output, i));

    sk_run_subgraph_forward(d.policy, actor->rank+1, critic->rank); // Run only the critic
  }

  /*
   * Compute the critic loss.
   */
  float critic_cost = sk_cost(critic, target_q, SK_QUADRATIC_COST);

  /*
   * Run the backward pass for the critic.
   */
  for(int i = 0; i < d.minibatch_size; i++)
    sk_run_subgraph_backward(d.policy, actor->rank+1, critic->rank);
  tensor_scalar_mul(d.policy->param_grad, 1.0f / d.minibatch_size); // Divide by N for mean

  /*
   * Update critic parameters.
   */
  d.optimizer.step(d.optimizer);

  /*
   * Compute the actor loss.
   */
  for(int i = 0; i < d.policy->depth; i++){ // Stop critic layers from calculating param grads
    if(d.policy->layers[i]->rank > actor->rank){
      d.policy->layers[i]->frozen = 1;
    }
  }

  /*
   * Run the backward pass for the actor.
   */
  for(int i = 0; i < d.minibatch_size; i++) // Run forward pass through entire network
    sk_forward(d.policy, minibatch[i].state);
  
  tensor_fill(critic->gradient, -1.0f); // Gradient ascent on critic
  sk_backward(d.policy);

  for(int i = 0; i < d.policy->depth; i++) // Unfreeze all layers
    d.policy->layers[i]->frozen = 0;

  /*
   * Update actor parameters
   */
  d.optimizer.step(d.optimizer);

  /*
   * Update target policy
   */
   tensor_scalar_mul(d.current_policy, 1 - d.tau); // (temporarily) multiply current parameters by 1-tau
   tensor_elementwise_add(d.target_policy, d.current_policy, d.target_policy); // Add these parameters to the target policy
   tensor_scalar_mul(d.current_policy, 1/(1 - d.tau)); // undo the 1-tau multiplication

  //printf("Critic cost is %f!\n", critic_cost / d.minibatch_size);
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

DDPG create_ddpg(Network *policy, size_t action_space, size_t state_space, size_t num_threads, size_t num_timesteps){
  DDPG d = {0};

  if(policy[0].is_recurrent)
    SK_ERROR("DDPG only works with ff policies.");

  d.policy = policy;
  d.policy_gradient = tensor_clone(SIEKNET_CPU, policy[0].param_grad);
  d.target_policy   = tensor_clone(SIEKNET_CPU, policy[0].params);
  d.current_policy  = tensor_clone(SIEKNET_CPU, policy[0].params);

  d._q_buffer = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 1);

  if(!policy)
    SK_ERROR("Received null ptr for policy.");

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
  d.tau      = 0.95;
  d.lr       = 1e-4;

  d.optimizer = create_optimizer(d.policy->params, d.policy->param_grad, SK_SGD);

  return  d;
}
