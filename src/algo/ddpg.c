#include <ddpg.h>
#include <math.h>
#include <time.h>

static size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

float ddpg_update_policy(DDPG d){
  size_t update_start = clock_us();
  if(d.n < d.minibatch_size)
    return 0.0f;

  /*
   * Sample from buffer
   */
  Transition minibatch[d.minibatch_size];
  for(int i = 0; i < d.minibatch_size; i++){
    int randidx = rand() % (d.n-1);
    minibatch[i] = d.replay_buffer[randidx];
  }

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
      float critic_q = tensor_at(d.critic_layer->output, i); // Get the critic value

      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward + d.discount * critic_q;
      //printf("Made %d %f\n", i, minibatch[i].reward + d.discount * critic_q);
    }else
      tensor_raw(target_q)[tensor_get_offset(target_q, i)] = minibatch[i].reward;
  }

  //tensor_print(target_q);
  //getchar();

  /*
   * Compute the current Q values.
   */
  //size_t q_comp_start = clock_us();
  tensor_copy(d.current_policy, d.policy->params); // Copy current parameters to policy
  d.policy->t = 0;
  for(int i = 0; i < d.minibatch_size; i++){
    tensor_copy(minibatch[i].action, get_subtensor(d.actor_layer->output, i));
    tensor_copy(minibatch[i].state, get_subtensor(d.state_layer->output, i));

    sk_run_subgraph_forward(d.policy, d.actor_layer->rank+1, d.critic_layer->rank); // Run only the critic
  }
  //float q_comp = (float)(clock_us() - q_comp_start)/1e6;

  /*
   * Compute the critic loss.
   */
  //size_t start = clock_us();
  float critic_cost = sk_cost(d.critic_layer, target_q, SK_QUADRATIC_COST);

  /*
   * Run the backward pass for the critic.
   */
  tensor_scalar_mul(d.critic_layer->gradient, 1.0f / d.minibatch_size); // Divide by N for mean
  for(int i = 0; i < d.minibatch_size; i++)
    sk_run_subgraph_backward(d.policy, d.actor_layer->rank+1, d.critic_layer->rank);

  //float elapsed = (float)(clock_us() - start)/1e6;
  //printf("elapsed: %f\n", elapsed);
  /*
   * Update critic parameters.
   */
  d.critic_optimizer.lr = d.critic_lr;
  d.critic_optimizer.step(d.critic_optimizer);

  /*
   * Compute the actor loss.
   */
  for(int i = 0; i < d.policy->depth; i++){ // Stop critic layers from calculating param grads
    if(d.policy->layers[i]->rank > d.actor_layer->rank){
      d.policy->layers[i]->frozen = 1;
    }
  }

  /*
   * Run the backward pass for the actor.
   */
  for(int i = 0; i < d.minibatch_size; i++) // Run forward pass through entire network
    sk_forward(d.policy, minibatch[i].state);
  
  tensor_fill(d.critic_layer->gradient, -1.0f / d.minibatch_size); // Gradient ascent on critic
  //tensor_scalar_mul(d.critic_layer->gradient, 1.0f / d.minibatch_size); // Divide by N for mean
  sk_backward(d.policy);

  /*
   * Update actor parameters
   */
  d.actor_optimizer.lr = d.actor_lr;
  d.actor_optimizer.step(d.actor_optimizer);

  for(int i = 0; i < d.policy->depth; i++) // Unfreeze all layers
    d.policy->layers[i]->frozen = 0;

  tensor_copy(d.policy->params, d.current_policy);

  /*
   * Update target policy
   */
   //size_t target_update_start = clock_us();
   tensor_scalar_mul(d.target_policy, 1 - d.tau); // multiply target parameters by 1-tau (value close to 1)
   tensor_scalar_mul(d.current_policy, d.tau); // (temporarily) multiply current parameters by tau (value close to 0)
   tensor_elementwise_add(d.target_policy, d.current_policy, d.target_policy); // Add current parameters to the target policy
   tensor_scalar_mul(d.current_policy, 1/d.tau); // undo the 1-tau multiplication
   return critic_cost / d.minibatch_size;
}

void ddpg_append_transition(DDPG *d, Tensor state, Tensor action, Tensor next_state, float reward, int terminal){
  size_t insert_point;
  if(d->n < d->num_timesteps-1){
    insert_point = d->n;
    d->n++;
  }else
    insert_point = rand() % d->num_timesteps;

  tensor_copy(state, d->replay_buffer[insert_point].state);
  tensor_copy(action, d->replay_buffer[insert_point].action);
  tensor_copy(next_state, d->replay_buffer[insert_point].next_state);

  d->replay_buffer[insert_point].reward = reward;
  d->replay_buffer[insert_point].terminal = terminal;
}

DDPG create_ddpg(Network *policy, size_t action_space, size_t state_space, size_t num_threads, size_t num_timesteps){
  if(!policy)
    SK_ERROR("Received null ptr for policy.");

  DDPG d = {0};
  d.policy = policy;

  if(d.policy->is_recurrent)
    SK_ERROR("DDPG only works with ff policies.");

  d.critic_layer = sk_layer_from_name(d.policy, "critic");
  if(!d.critic_layer)
    SK_ERROR("Unable to find layer with name 'critic'");

  d.actor_layer = sk_layer_from_name(d.policy, "actor");
  if(!d.actor_layer)
    SK_ERROR("Unable to find layer with name 'actor'");

  d.state_layer = sk_layer_from_name(d.policy, "state");
  if(!d.state_layer)
    SK_ERROR("Unable to find layer with name 'state'");

  d.target_policy   = tensor_clone(SIEKNET_CPU, policy[0].params);
  d.current_policy  = tensor_clone(SIEKNET_CPU, policy[0].params);

  size_t actor_param_boundary  = d.actor_layer->param_idx + d.actor_layer->num_params;
  size_t critic_param_boundary = d.critic_layer->param_idx + d.critic_layer->num_params;

  d.actor_params = get_subtensor_reshape(policy->params, 0, actor_param_boundary);
  d.critic_params = get_subtensor_reshape(policy->params, actor_param_boundary, critic_param_boundary - actor_param_boundary);

  d.actor_param_grad = get_subtensor_reshape(policy->param_grad, 0, actor_param_boundary);
  d.critic_param_grad = get_subtensor_reshape(policy->param_grad, actor_param_boundary, critic_param_boundary - actor_param_boundary);

  d._q_buffer = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, 1);

  d.replay_buffer = (Transition *)malloc(sizeof(Transition) * num_timesteps);
  for(int i = 0; i < num_timesteps; i++){
    d.replay_buffer[i].state      = create_tensor(SIEKNET_CPU, state_space);
    d.replay_buffer[i].next_state = create_tensor(SIEKNET_CPU, state_space);
    d.replay_buffer[i].action     = create_tensor(SIEKNET_CPU, action_space);
    d.replay_buffer[i].reward     = -INFINITY;
    d.replay_buffer[i].terminal   = 0;
  }
  
  d.n = 0;
  d.minibatch_size = 64;
  d.num_timesteps = num_timesteps;

  d.discount = 0.99;
  d.tau      = 1e-3;
  d.actor_lr  = 1e-4;
  d.critic_lr = 1e-3;

  d.actor_optimizer  = create_optimizer(d.actor_params, d.actor_param_grad, SK_SGD);
  d.critic_optimizer = create_optimizer(d.critic_params, d.critic_param_grad, SK_SGD);


  return  d;
}
