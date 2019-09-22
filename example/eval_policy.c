#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>
#include <locale.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>
#include <cgym.h>
#include <ars.h>

float rollout(Network *n, Environment *e, float shift, int max_traj_len){

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
    e->render(*e);

  } while(!*e->done && n->t < max_traj_len);

  tensor_dealloc(x);
  n->t = 0;
  sk_wipe(n);
  return reward;
}

int main(int argc, char **argv){
  char *model_path       = NULL;
  char *weight_path      = NULL;
  char *environment_name = NULL;
  
  size_t random_seed = time(NULL);
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
      {"traj_len",        required_argument, 0,  0},
      {0,                 0,                 0,  0},
    };

    int opt_idx;
    char c = getopt_long_only(argc, argv, "", long_options, &opt_idx);
    if(!c){
      if(!strcmp(long_options[opt_idx].name, "model"))     model_path  = optarg;
      if(!strcmp(long_options[opt_idx].name, "weights"))   weight_path = optarg;
      if(!strcmp(long_options[opt_idx].name, "env"))       environment_name = optarg;
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

  Network n;
  Environment e;

  /* Create an identical network for every thread */
  if(weight_path)
    SK_ERROR("Need to implement this!"); //TODO
  else
    n = sk_create(model_path);

  /* Create identical environments for every thread */
#ifdef COMPILED_FOR_MUJOCO
  if(!strcmp(environment_name, "humanoid"))
    e = create_humanoid_env();
#else
  if(!strcmp(environment_name, "cassie"))
    e = create_cassie_env();
#endif

  printf("\n   _____ ____________ __ _   ______________  \n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/  \n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /      \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	     \n");
  printf("																					   \n");
  printf("Evaluate and visualize RL policies\n\n");
  printf("Environment: '%s'\n", environment_name);
  printf("Policy:      '%s'\n", model_path);
  printf("\n");

  srand(random_seed);

  size_t iter = 0;
  float avg_return = 0;
  while(1){

    float reward = rollout(&n, &e, 0.0f, max_traj_len);
    avg_return += reward;

    float avg = avg_return / (iter++ + 1);

    printf("Return: %5.2f, avg %5.2f\t\n", reward, avg);
  }
}

