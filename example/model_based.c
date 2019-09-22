/* DEPRECATED */

#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>
#include <cgym.h>

size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

#if 0
float rollout(Network *n, Environment *e, int do_grads){
  Layer *out, *state_pred, *reward_pred, *input_embedding;
  Tensor reward_label = {0};
  Tensor state_label = {0};

  out = sk_layer_from_name(n, "action");
  if(!out)
    SK_ERROR("Model has no layer with name 'action'.\n");

  if(do_grads){
    state_pred = sk_layer_from_name(n, "state_prediction");
    if(!state_pred)
      SK_ERROR("Model has no layer with name 'state_prediction'\n");

    reward_pred = sk_layer_from_name(n, "reward_prediction");
    if(!reward_pred)
      SK_ERROR("Model has no layer with name 'reward_prediction'\n");
  }

  if(out->size != e->action_space)
    SK_ERROR("Model action layer doesn't match environment action space (%lu vs %lu)\n", out->size, e->action_space);

  if(n->input_dimension != e->observation_space)
    SK_ERROR("Model input dimension doesn't match environment observation space (%lu vs %lu)\n", n->input_dimension, e->observation_space);

  Tensor x = create_tensor(SIEKNET_CPU, n->input_dimension);
  e->reset(*e);
  e->seed(*e);

  float buff[out->size];
  do {
    for(int j = 0; j < n->input_dimension; j++)
      tensor_raw(x)[tensor_get_offset(x, j)] = e->state[j];

    sk_forward(n, x);
    Tensor current_out = get_subtensor(out->output, n->t);

    for(int j = 0; j < out->size; j++)
      buff[j] = tensor_at(current_out, j);

    e->step(*e, buff);
    e->render(*e);
  } while(!*e->done && n->t < 400);

  tensor_dealloc(x);

  return 0.0f;
}
#endif

int main(int argc, char **argv){
  char *model_path        = NULL;
  char *weight_path       = NULL;
  char *environment_name  = NULL;

  setbuf(stdout, NULL);

  /*
   * Read command line options
   */
  int args_read = 0;
  while(1){
    static struct option long_options[] = {
      {"model",           required_argument, 0,  0 },
      {"weights",         required_argument, 0,  0 },
      {"env",             required_argument, 0,  0 },
      {0,                 0,                 0,  0 },
    };

    int opt_idx;
    char c = getopt_long_only(argc, argv, "", long_options, &opt_idx);
    if(!c){
      if(!strcmp(long_options[opt_idx].name, "model"))   model_path  = optarg;
      if(!strcmp(long_options[opt_idx].name, "weights")) weight_path = optarg;
      if(!strcmp(long_options[opt_idx].name, "env"))     environment_name = optarg;
      args_read++;
    }else if(c == -1) break;
  }

  int success = 1;
  if(args_read < 3){
    if(!model_path){
      printf("Missing arg: --model [.sk file]\n");
      success = 0;
    }
    if(!environment_name){
      printf("Missing arg: --env [envname]\n");
      success = 0;
    }
    if(!success)
      exit(1);
  }

  Network n;
  if(weight_path)
    asm("nop"); //TODO: load weights
  else
    n = sk_create(model_path);

  printf("Model: '%s'\n", n.name);
  for(int i = 0; i < n.depth; i++){
    printf("\tExecution rank %d: '%s'\n", n.layers[i]->rank, n.layers[i]->name);
    printf("\t\tParam offset: %lu\n", n.layers[i]->param_idx);
    printf("\t\tParams:       %lu\n", n.layers[i]->num_params);
  }

  Environment e;
  if(!strcmp(environment_name, "ant"))
    e = create_ant_env();
  if(!strcmp(environment_name, "hopper"))
    e = create_hopper_env();
  if(!strcmp(environment_name, "half_cheetah"))
    e = create_hopper_env();
  if(!strcmp(environment_name, "humanoid"))
    e = create_humanoid_env();
  if(!strcmp(environment_name, "walker2d"))
    e = create_walker2d_env();

  if(!strcmp(environment_name, "cassie"))
    e = create_cassie_env();

  Layer *action, *state_pred, *reward_pred, *input_embedding;

  action = sk_layer_from_name(&n, "action");
  if(!action)
    SK_ERROR("Model has no layer with name 'action'.\n");

  state_pred = sk_layer_from_name(&n, "state_prediction");
  if(!state_pred)
    SK_ERROR("Model has no layer with name 'state_prediction'\n");

  reward_pred = sk_layer_from_name(&n, "reward_prediction");
  if(!reward_pred)
    SK_ERROR("Model has no layer with name 'reward_prediction'\n");

  input_embedding = sk_layer_from_name(&n, "input_embedding");
  if(!input_embedding)
    SK_ERROR("Model has no layer with name 'input_embedding'\n");

  Tensor reward_label = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, reward_pred->size);
  Tensor state_label  = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, state_pred->size);
  Tensor max_reward   = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, reward_pred->size);
  Tensor adjust_grad  = create_tensor(SIEKNET_CPU, SIEKNET_MAX_UNROLL_LENGTH, reward_pred->size);

  float best_reward = -INFINITY;

  Tensor x = create_tensor(SIEKNET_CPU, n.input_dimension);
  float buff[action->size];

  Optimizer o = create_optimizer(n.params, n.param_grad, SK_SGD);

  size_t episodes = 1000;
  size_t rollouts = 100;
  size_t steps = 0;
  
  double reward_mean = 0;
  double reward_mean_diff = 0;
  for(int i = 0; i < episodes; i++){
    e.reset(e);
    e.seed(e);
    n.t = 0;
    float episode_reward = 0;
    float episode_reward_cost = 0;
    float episode_state_cost = 0;
    for(int k = 0; k < rollouts; k++){
      float rollout_reward = 0;
      do {
        int t = n.t;

        for(int j = 0; j < n.input_dimension; j++)
          tensor_raw(x)[tensor_get_offset(x, j)] = e.state[j];

        sk_forward(&n, x);

        Tensor current_out = get_subtensor(action->output, t);
        for(int j = 0; j < action->size; j++)
          buff[j] = tensor_at(current_out, j);

        float true_reward = e.step(e, buff);
        rollout_reward += true_reward;

        Tensor state_label_t = get_subtensor(state_label, t);
        for(int j = 0; j < n.input_dimension; j++)
          tensor_raw(state_label_t)[tensor_get_offset(state_label_t, j)] = e.state[j];

        Tensor reward_label_t = get_subtensor(reward_label, t);
        tensor_raw(reward_label_t)[tensor_get_offset(reward_label_t, 0)] = true_reward;

        if(true_reward > best_reward)
          best_reward = true_reward;

        //e.render(e);
      } while(!*e.done && n.t < 400);

      size_t ep_len = n.t;
      reward_label.dims[0] = ep_len;
      state_label.dims[0]  = ep_len;
      max_reward.dims[0]   = ep_len;
      adjust_grad.dims[0]  = ep_len;

      tensor_fill(max_reward, best_reward);

      // Do gradient calc for the state prediction
      float state_cost = sk_cost(state_pred, state_label, SK_QUADRATIC_COST);
      for(int j = 0; j < n.depth; j++){
        if(/*n.layers[j] == input_embedding || */n.layers[j] == state_pred){
          n.layers[j]->frozen = 0;
          n.layers[j]->blocking = 0;
        }else{
          n.layers[j]->frozen = 1;
          n.layers[j]->blocking = 1;
        }
      }
      sk_backward(&n);
      n.t = ep_len;

      // Do gradient calc for reward prediction
      float reward_cost = sk_cost(reward_pred, reward_label, SK_QUADRATIC_COST);
      for(int j = 0; j < n.depth; j++){
        if(/*n.layers[j] == input_embedding || */n.layers[j] == reward_pred){
          n.layers[j]->frozen = 0;
        }else{
          n.layers[j]->frozen = 1;
        }
        n.layers[j]->blocking = 0;
      }
      sk_backward(&n);
      n.t = ep_len;

#if 1
      // Do gradient ascent on approximated reward
      //float ascent_cost = sk_cost(reward_pred, max_reward, SK_QUADRATIC_COST);
      tensor_fill(reward_pred->gradient, -1.0f);
      for(int j = 0; j < n.depth; j++){
        if(n.layers[j] == reward_pred){
          n.layers[j]->frozen = 1;
          n.layers[j]->blocking = 0;
        }else if(n.layers[j] == action || n.layers[j] == input_embedding){
          n.layers[j]->frozen = 0;
          n.layers[j]->blocking = 1;
        }else{
          n.layers[j]->frozen = 1;
          n.layers[j]->blocking = 1;
        }
      }

      float std = 1; //TODO: std of rewards

      tensor_elementwise_sub(reward_label, reward_pred->output, adjust_grad);

      tensor_fabs(adjust_grad);
      tensor_scalar_mul(adjust_grad, -1.0f / std);
      tensor_expf(adjust_grad);
      tensor_elementwise_mul(adjust_grad, reward_pred->gradient, reward_pred->gradient);
      tensor_scalar_mul(reward_pred->gradient, -1.0f);

      sk_backward(&n);
      n.t = ep_len;

      episode_reward      += rollout_reward / ep_len;
      episode_state_cost  += state_cost / ep_len;
      episode_reward_cost += reward_cost / ep_len;
      steps += ep_len;
    }
    o.lr = 1e-6;
#endif
    printf("Ep. %d, reward %f, state cost %f, reward cost %f, steps %'9lu\n", i+1, episode_reward / episodes, episode_state_cost / episodes, episode_reward_cost / episodes, steps);
    if(!(i%10)){
      o.step(o);
    }
    sk_wipe(&n);
  }
  printf("done!\n");


  //rollout(&n, &e);

  /*
   */

}
