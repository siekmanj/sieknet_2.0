#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>
#include <locale.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>
#include <cgym.h>
#include <ddpg.h>

size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

int main(int argc, char **argv){
  char *model_path       = NULL;
  char *weight_path      = NULL;
  char *environment_name = NULL;
  
  size_t num_threads = 1;
  size_t num_iterations = 100;
  size_t random_seed = time(NULL);
  size_t timesteps = 1e7;

  float step_size = 0.02f;
  float gamma = 0.99;

  size_t max_traj_len = 400;

  setbuf(stdout, NULL);
  setlocale(LC_ALL,"");

  /*
   * Read command line options
   */
  int args_read = 0;
  while(1){
    static struct option long_options[] = {
      {"model",           required_argument, 0,  0},
      {"weights",         required_argument, 0,  0},
      {"env",             required_argument, 0,  0},
      {"threads",         required_argument, 0,  0},
      {"seed",            required_argument, 0,  0},
      {"step_size",       required_argument, 0,  0},
      {"gamma",           required_argument, 0,  0},
      {"timesteps",       required_argument, 0,  0},
      {"traj_len",        required_argument, 0,  0},
      {0,                 0,                 0,  0},
    };

    int opt_idx;
    char c = getopt_long_only(argc, argv, "", long_options, &opt_idx);
    if(!c){
      if(!strcmp(long_options[opt_idx].name, "model"))     model_path  = optarg;
      if(!strcmp(long_options[opt_idx].name, "weights"))   weight_path = optarg;
      if(!strcmp(long_options[opt_idx].name, "env"))       environment_name = optarg;
      if(!strcmp(long_options[opt_idx].name, "threads"))   num_threads = strtol(optarg, NULL, 10);
      if(!strcmp(long_options[opt_idx].name, "seed"))      random_seed = strtol(optarg, NULL, 10);
      if(!strcmp(long_options[opt_idx].name, "step_size")) step_size = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "gamma"))     gamma = strtol(optarg, NULL, 10);
      if(!strcmp(long_options[opt_idx].name, "timesteps")) timesteps = (size_t)strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "traj_len"))  max_traj_len = strtol(optarg, NULL, 10);
      args_read++;
    }else if(c == -1) break;
  }

  int success = 1;
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

  int load_weights = 0;
  if(weight_path){
    FILE *fp = fopen(weight_path, "rb");
    if(!fp)
      load_weights = 0;
    else{
      load_weights = 1;
      fclose(fp);
    }
  }

  Network n = {0};
  if(!load_weights)
    n = sk_create(model_path);

  Environment env;
  if(!strcmp(environment_name, "ant"))
    env = create_ant_env();
  if(!strcmp(environment_name, "hopper"))
    env = create_hopper_env();
  if(!strcmp(environment_name, "half_cheetah"))
    env = create_hopper_env();
  if(!strcmp(environment_name, "humanoid"))
    env = create_humanoid_env();
  if(!strcmp(environment_name, "walker2d"))
    env = create_walker2d_env();

  //TODO LOAD POLICY
  //TODO LOAD ENV

  printf("Creating algo!\n");
  DDPG algo = create_ddpg(&n, env.action_space, env.observation_space, 1, 1e7);
  algo.discount = gamma;
  algo.minibatch_size = 500;

  size_t reset_every = 50;

  Layer *out = sk_layer_from_name(&n, "actor");

  if(!out)
    SK_ERROR("Could not find layer with name 'action'");

  Tensor state_t = create_tensor(SIEKNET_CPU, env.observation_space);
  Tensor next_state = create_tensor(SIEKNET_CPU, env.observation_space);
  float action_buff[env.action_space];
  size_t iter = 0;
  float avg_return = 0;
  while(1){
    size_t start = clock_us();
    if(!(iter % reset_every)){
      avg_return = 0;
      printf("\n");
    }
    /*
     * Gather samples for this iteration.
     */
    env.reset(env);
    env.seed(env);
    n.t = 0;
    tensor_copy(algo.current_policy, n.params);
    do {
      memset(action_buff, '\0', sizeof(float)*env.action_space);
      tensor_fill(state_t, 0.0f);

      for(int i = 0; i < env.observation_space; i++)
        tensor_raw(state_t)[tensor_get_offset(state_t, i)] = env.state[i];

      sk_forward(&n, state_t);
      Tensor action = get_subtensor(out->output, n.t-1);

      for(int i = 0; i < env.action_space; i++)
        action_buff[i] = tensor_at(action, i);

      float r = env.step(env, action_buff);

      if(!*env.done){
        for(int i = 0; i < env.observation_space; i++)
          tensor_raw(next_state)[tensor_get_offset(state_t, i)] = env.state[i];
      }else
        tensor_fill(next_state, 0.0f);
      
      /*
       * Append this transition to the replay buffer.
       */
      ddpg_append_transition(&algo, state_t, action, next_state, r, *env.done);

    } while(!*env.done && n.t < max_traj_len);

#if 0
    printf("******************************\nstate of buffer:\n");
    for(int i = 0; i < algo.n; i++){
      printf("\tPOS %d:\n", i);
      tensor_print(algo.replay_buffer[i].state);
      tensor_print(algo.replay_buffer[i].action);
      printf("REWARD %f\n", algo.replay_buffer[i].reward);
      printf("TERMINAL %d\n", algo.replay_buffer[i].terminal);
    }
    getchar();
#endif

  /*
   * Update policy here
   */
    //printf("Updating policy.\n");
    ddpg_update_policy(algo);

   /*
    * Evalulate policy
    */ 
#if 1
    tensor_copy(algo.target_policy, n.params);
    float reward = 0.0f;
    do {
     memset(action_buff, '\0', sizeof(float)*env.action_space);
     tensor_fill(state_t, 0.0f);

     for(int i = 0; i < env.observation_space; i++)
       tensor_raw(state_t)[tensor_get_offset(state_t, i)] = env.state[i];

     sk_forward(&n, state_t);
     Tensor action = get_subtensor(out->output, n.t-1);

     for(int i = 0; i < env.action_space; i++)
       action_buff[i] = tensor_at(action, i);

     reward += env.step(env, action_buff);
    }while(!*env.done && n.t < max_traj_len);
    //printf("return %f\n", reward);
#endif
    avg_return += reward;

    float batch_avg = avg_return / ((iter % reset_every) + 1);
    double elapsed = (clock_us() - start)/1e6;
    printf("Iteration %lu took %3.2fs | avg return over last %lu iterations %f\t\r", iter+1, elapsed, (iter % reset_every) + 1, batch_avg);
    iter++;
  }
}
