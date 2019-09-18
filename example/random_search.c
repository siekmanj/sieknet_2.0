#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>
#include <locale.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>
#include <cgym.h>
#include <ars.h>

size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

float rollout(Network *n, Environment *e, size_t *num_steps, int render, float shift, int max_traj_len){

  Layer *out = sk_layer_from_name(n, "action");
  if(!out)
    SK_ERROR("Model has no layer with name 'action'.\n");

  if(out->size != e->action_space)
    SK_ERROR("Model action layer doesn't match environment action space (%lu vs %lu)\n", out->size, e->action_space);

  if(n->input_dimension != e->observation_space)
    SK_ERROR("Model input dimension doesn't match environment observation space (%lu vs %lu)\n", n->input_dimension, e->observation_space);

  Tensor x = create_tensor(SIEKNET_CPU, n->input_dimension);
  e->reset(*e);
  e->seed(*e);

  size_t rollout_steps = 0;
  float reward = 0;
  float buff[out->size];
  do {
    for(int j = 0; j < n->input_dimension; j++)
      tensor_raw(x)[tensor_get_offset(x, j)] = e->state[j];

    sk_forward(n, x);
    Tensor current_out = get_subtensor(out->output, n->t);

    for(int j = 0; j < out->size; j++)
      buff[j] = tensor_at(current_out, j);

    reward += e->step(*e, buff) - shift;
    if(render)
      e->render(*e);

    rollout_steps++;
  } while(!*e->done && n->t < max_traj_len);

  if(num_steps)
    *num_steps = rollout_steps;

  tensor_dealloc(x);
  n->t = 0;
  sk_wipe(n);
  return reward;
}

Network *networks;
Environment *envs;
size_t *steps;
size_t max_traj_len;

float R(Tensor input, size_t thread_num){
  size_t rollout_timesteps = 0;
  tensor_copy(input, networks[thread_num].params);
  float reward = rollout(&networks[thread_num], &envs[thread_num], &rollout_timesteps, 0, envs[thread_num].alive_bonus, max_traj_len);
  steps[thread_num] += rollout_timesteps;
  return reward;
}

int main(int argc, char **argv){
  char *model_path       = NULL;
  char *weight_path      = NULL;
  char *environment_name = NULL;
  
  size_t num_threads = 1;
  size_t num_iterations = 100;
  size_t random_seed = time(NULL);
  size_t num_deltas = 10;
  size_t timesteps = 1e5;

  float step_size = 0.02f;
  float std_dev = 0.0075f;

  max_traj_len = 400;

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
      {"std",             required_argument, 0,  0},
      {"deltas",          required_argument, 0,  0},
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
      if(!strcmp(long_options[opt_idx].name, "std"))       std_dev = strtof(optarg, NULL);
      if(!strcmp(long_options[opt_idx].name, "deltas"))    num_deltas = strtol(optarg, NULL, 10);
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

  Network     stack_nets[num_threads];
  Environment stack_envs[num_threads];
  size_t      stack_steps[num_threads];

  envs     = stack_envs;
  networks = stack_nets;
  steps    = stack_steps;

  memset(steps, '\0', sizeof(steps[0]) * num_threads);

  /* Create an identical network for every thread */
  if(weight_path)
    SK_ERROR("Need to implement this!"); //TODO
  else{
    size_t start = clock_us();
    for(int i = 0; i < num_threads; i++){
      networks[i] = sk_create_network(model_path);
      printf("Loaded %3d of %3lu networks in %5.4f seconds.\r", i+1, num_threads, (clock_us() - start)/1e6);
    }
  }
  printf("\n");

  /* Create identical environments for every thread */
  if(!strcmp(environment_name, "ant"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_ant_env();
  if(!strcmp(environment_name, "hopper"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_hopper_env();
  if(!strcmp(environment_name, "half_cheetah"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_hopper_env();
  if(!strcmp(environment_name, "humanoid"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_humanoid_env();
  if(!strcmp(environment_name, "walker2d"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_walker2d_env();

  if(!strcmp(environment_name, "cassie"))
    for(int i = 0; i < num_threads; i++)
      envs[i] = create_cassie_env();

  printf("\n   _____ ____________ __ _   ______________  \n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/  \n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /      \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	     \n");
  printf("																					   \n");
  printf("Augmented Random Search for reinforcement learning demo\n\n");
  printf("Environment: '%s'\n", environment_name);
  printf("Policy:      '%s'\n", model_path);
  printf("Iterations:  %'lu\n", num_iterations);
  printf("Threads:     %lu\n", num_threads);
  printf("Random seed: %lu\n", random_seed);
  printf("Step size:   %g\n", step_size);
  printf("Std. dev:    %g\n", std_dev);
  printf("Deltas:      %'lu\n", num_deltas);
  printf("Timesteps:   %'lu\n", timesteps);
  printf("\n");

  srand(random_seed);

  ARS algo = create_ars(R, networks[0].params, num_deltas, num_threads);

  algo.step_size = step_size;
  algo.std = std_dev;

  //TODO: zero weight init option in .sk config
  tensor_fill(algo.params, 0.0f);

  size_t steps_before = 0;
  size_t num_steps;
  size_t iter = 0;

  float avg_secs_per_sample = 0.0f;
  float avg_trend           = 0.0f;
  float avg_return          = 0.0f;
  float last_reward         = 0.0f;

  size_t reset_every = 50;
  do {
    size_t start = clock_us();
    algo.step(algo);
    double elapsed = (clock_us() - start)/1e6;

    num_steps = 0;
    for(int j = 0; j < num_threads; j++)
      num_steps += steps[j];

    if(!(iter % reset_every)){
      avg_return = 0;
      avg_trend = 0;
      printf("\n");
    }

    float secs_per_sample = elapsed / (num_steps - steps_before);
    tensor_copy(algo.params, networks[0].params);
    float reward = rollout(&networks[0], &envs[0], NULL, 0, 0.0f, max_traj_len);
    avg_return += reward;

    if(iter == 1)
      last_reward = reward;
    avg_trend += reward - last_reward;

    avg_secs_per_sample += secs_per_sample;
    float completion      = (double)num_steps / (double)timesteps;
    float time_left       = ((1 - completion) * timesteps) * (avg_secs_per_sample/(iter+1));
    time_left = time_left > 0 ? time_left : 0;

    int hrs_left = (int)(time_left / (60 * 60));
    int min_left = ((int)(time_left - (hrs_left * 60 * 60))) / 60;
    int sec_left = (int)(time_left - (hrs_left * 60 * 60 + min_left * 60));

    float batch_avg = avg_return / ((iter % reset_every) + 1);
    float batch_trend = avg_trend / ((iter % reset_every) + 1);
    printf("Iteration %lu took %3.2fs | avg return %6.2f | trend %6.2f | %5.3fs per 1k timesteps | %3dh %2dm %2ds remain | %'9lu \t\r", iter+1, elapsed, batch_avg, batch_trend, secs_per_sample * 1000, hrs_left, min_left, sec_left, num_steps);

    steps_before = num_steps;
    last_reward = reward;
    iter++;
  }
  while(num_steps < timesteps);
  printf("\nExperiment concluded.\n");
}
