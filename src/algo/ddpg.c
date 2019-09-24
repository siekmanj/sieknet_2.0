#include <ddpg.h>
#include <math.h>

void ddpg_update_policy(DDPG d){
  if(d.n < d.minibatch_size)
    return;

  /*
   * Sample from buffer
   */
  Transition minibatch[d.minibatch_size];
  //printf("****************************\n");
  for(int i = 0; i < d.minibatch_size; i++){
    int randidx = rand() % (d.n-1);
    minibatch[i] = d.transitions[randidx];

    //printf("sampled transition %d\n", randidx);
    //tensor_print(minibatch[i].state);
    //tensor_print(minibatch[i].action);
    //printf("terminal state: %d\n", minibatch[i].terminal);
    //printf("reward: %f\n", minibatch[i].reward);
  }
  //getchar();

  /*
   *
   */

  /*
   *
   */

  /*
   *
   */
}

void ddpg_append_transition(DDPG *d, Tensor state, Tensor action, float reward, int terminal){
  if(d->n < d->num_timesteps-1){
    tensor_copy(state, d->transitions[d->n].state);
    tensor_copy(action, d->transitions[d->n].action);
    d->transitions[d->n].reward = reward;
    d->transitions[d->n].terminal = terminal;

#if 0
    printf("transition %lu of %lu:\n", d->n, d->num_timesteps-1);
    tensor_print(d->transitions[d->n].action);
    tensor_print(action);
#endif

    d->n++;
  }else
    SK_ERROR("TODO: Handle this case.\n");
}

DDPG create_ddpg(Network *networks, size_t action_space, size_t state_space, size_t num_threads, size_t num_timesteps){
  DDPG d = {0};

  if(networks[0].is_recurrent)
    SK_ERROR("DDPG only works with ff policies.");

  d.policy_gradient = tensor_clone(SIEKNET_CPU, networks[0].param_grad);
  d.target_policy = tensor_clone(SIEKNET_CPU, networks[0].params);
  d.current_policy = tensor_clone(SIEKNET_CPU, networks[0].params);

  if(!networks)
    SK_ERROR("Received null ptr for networks.");

  d.transitions = (Transition *)malloc(sizeof(Transition) * num_timesteps);
  for(int i = 0; i < num_timesteps; i++){
    d.transitions[i].state  = create_tensor(SIEKNET_CPU, state_space);
    d.transitions[i].action = create_tensor(SIEKNET_CPU, action_space);
    d.transitions[i].reward = -INFINITY;
    d.transitions[i].terminal = 0;
  }
  
  d.n = 0;
  d.minibatch_size = 16;
  d.num_timesteps = num_timesteps;

  printf("BEFORE RETRUN\n");
  tensor_print(d.target_policy);

  return  d;
}
