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
  
  size_t num_threads       = 1;
  size_t num_iterations    = 100;
  size_t random_seed       = time(NULL);
  size_t timesteps         = 1e6;
  size_t minibatch_size    = 256;
  size_t max_traj_len      = 400;
  size_t steps             = 0;
  size_t eval_frequency    = 1e3;

  float tau              = 0.005;
  float gamma            = 0.99;
  float action_std       = 0.2;
  float alpha            = 1.0f;

  float actor_step_size  = 1e-4;
  float critic_step_size = 1e-3;

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
      {"gamma",           required_argument, 0,  0},
      {"tau",             required_argument, 0,  0},
      {"actor_lr",        required_argument, 0,  0},
      {"critic_lr",       required_argument, 0,  0},
      {"timesteps",       required_argument, 0,  0},
      {"traj_len",        required_argument, 0,  0},
      {"minibatch",       required_argument, 0,  0},
      {"std",             required_argument, 0,  0},
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
      if(!strcmp(long_options[opt_idx].name, "gamma"))     gamma = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "tau"))       tau = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "actor_lr"))  actor_step_size = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "critic_lr")) critic_step_size = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "std"))       action_std = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "timesteps")) timesteps = (size_t)strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "traj_len"))  max_traj_len = strtol(optarg, NULL, 10);
      if(!strcmp(long_options[opt_idx].name, "minibatch")) minibatch_size = strtol(optarg, NULL, 10);
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

  Environment env;
  if(!environment_name)
    SK_ERROR("Env name not provided.");
  else if(!strcmp(environment_name, "ant"))
    env = create_ant_env();
  else if(!strcmp(environment_name, "hopper"))
    env = create_hopper_env();
  else if(!strcmp(environment_name, "half_cheetah"))
    env = create_half_cheetah_env();
  else if(!strcmp(environment_name, "humanoid"))
    env = create_humanoid_env();
  else if(!strcmp(environment_name, "walker2d"))
    env = create_walker2d_env();
  else
    SK_ERROR("Invalid env '%s'", environment_name);

  srand(random_seed);

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
   

  printf("Creating algo!\n");
  DDPG algo = create_ddpg(&n, env.action_space, env.observation_space, 1, timesteps);
  algo.discount = gamma;
  algo.minibatch_size = minibatch_size;
  algo.actor_lr = actor_step_size;
  algo.critic_lr = critic_step_size;
  algo.tau = tau;
  algo.num_timesteps = timesteps;

  printf("\n   _____ ____________ __ _   ______________  \n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/  \n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /      \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	     \n");
  printf("																					   \n");
  printf("Deep Deterministic Policy Gradients for reinforcement learning demo\n\n");
  printf("Environment:    '%s'\n", environment_name);
  printf("Policy:         '%s'\n", model_path);
  printf("Iterations:      %'lu\n", num_iterations);
  printf("Threads:         %lu\n", num_threads);
  printf("Random seed:     %lu\n", random_seed);
  printf("Actor lr:        %g\n", algo.actor_lr);
  printf("Critic lr:       %g\n", algo.critic_lr);
  printf("Tau:             %g\n", algo.tau);
  printf("Discount:        %g\n", algo.discount);
  printf("Reward factor:   %g\n", alpha);
  printf("Batch size:      %'lu\n", algo.minibatch_size);
  printf("Buffer size:     %'lu\n", algo.num_timesteps);
  printf("\n");

  size_t reset_every = 10;

  Layer *out = sk_layer_from_name(&n, "actor");

  if(!out)
    SK_ERROR("Could not find layer with name 'actor'");

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
    //tensor_copy(algo.current_policy, n.params);
    size_t samples_gathered = 0;
    float critic_cost = 0;
    //while(samples_gathered < samples_per_iter){
      env.reset(env);
      env.seed(env);
      n.t = 0;
      do {
        memset(action_buff, '\0', sizeof(float)*env.action_space);
        tensor_fill(state_t, 0.0f);

        for(int i = 0; i < env.observation_space; i++)
          tensor_raw(state_t)[tensor_get_offset(state_t, i)] = env.state[i];

        sk_forward(&n, state_t);
        Tensor action = get_subtensor(out->output, n.t-1);

        for(int i = 0; i < action.size; i++)
          tensor_raw(action)[tensor_get_offset(action, i)] += normal(0, action_std);

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
        ddpg_append_transition(&algo, state_t, action, next_state, alpha*r, *env.done);

        critic_cost += ddpg_update_policy(algo);

        printf("Did %5lu timesteps in %3.2fs\r", samples_gathered, (clock_us() - start)/1e6);

        samples_gathered++;

      } while(!*env.done && n.t < max_traj_len);
    //}
    steps += samples_gathered;
    critic_cost /= samples_gathered;

   /*
    * Evalulate policy
    */ 
    tensor_copy(algo.current_policy, n.params);
    float reward = 0.0f;
    n.t = 0;
    env.reset(env);
    env.seed(env);
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
     //if(!(iter % reset_every))
       env.render(env);

    }while(!*env.done && n.t < max_traj_len);
    avg_return += reward;

    float batch_avg = avg_return / ((iter % reset_every) + 1);
    double elapsed = (clock_us() - start)/1e6;

    float efficiency = samples_gathered / elapsed;
    printf("Iteration %4lu took %5.1fs | r %6.2f | last %3lu: %6.2f | cost: %f | %4.1f fps | timesteps %'9lu\t\r", iter+1, elapsed, reward, (iter % reset_every) + 1, batch_avg, critic_cost, efficiency, steps);
    printf("\n");
    iter++;
  }

  //tensor_copy(algo.target_policy, n.params);
  while(1){
    n.t = 0;
    env.reset(env);
    env.seed(env);
    do {
     memset(action_buff, '\0', sizeof(float)*env.action_space);
     tensor_fill(state_t, 0.0f);

     for(int i = 0; i < env.observation_space; i++)
       tensor_raw(state_t)[tensor_get_offset(state_t, i)] = env.state[i];

     sk_forward(&n, state_t);
     Tensor action = get_subtensor(out->output, n.t-1);

     for(int i = 0; i < env.action_space; i++)
       action_buff[i] = tensor_at(action, i);

     env.step(env, action_buff);

     env.render(env);

    }while(!*env.done && n.t < max_traj_len);
  }
}
