#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>
#include <cgym.h>
#include <ars.h>

#ifndef NUM_THREADS
#define NUM_THREADS 10
#endif

size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

float rollout(Network *n, Environment *e){

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

Network networks[NUM_THREADS];
Environment envs[NUM_THREADS];

float R(Tensor input, size_t thread_num){
  tensor_copy(input, networks[thread_num].params);
  return rollout(&networks[thread_num], &envs[thread_num]);
}

int main(int argc, char **argv){
  char *model_path       = NULL;
  char *weight_path      = NULL;
  char *environment_name = NULL;

  setbuf(stdout, NULL);

  /*
   * Read command line options
   */
  int args_read = 0;
  while(1){
    static struct option long_options[] = {
      {"model",           required_argument, 0,  0},
      {"weights",         required_argument, 0,  0},
      {"env",             required_argument, 0,  0},
      {0,                 0,                 0,  0},
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

  if(weight_path)
    SK_ERROR("Need to implement this!");
  else
    for(int i = 0; i < NUM_THREADS; i++)
      networks[i] = sk_create_network(model_path);

#ifdef COMPILED_FOR_MUJOCO
  if(!strcmp(environment_name, "humanoid"))
    for(int i = 0; i < NUM_THREADS; i++)
      envs[i] = create_humanoid_env();
#else
  if(!strcmp(environment_name, "cassie"))
    for(int i = 0; i < NUM_THREADS; i++)
      envs[i] = create_cassie_env();
#endif

  ARS algo = create_ars(R, networks[0].params, 10, 1);
  //for(int i = 0; i < NUM_THREADS; i++){
  //  rollout(&networks[i], &envs[i]);
  //}

  /*
   * TODO: Environment stuff
   * TODO: State normalization (part of model or part of env?)
   * TODO: ARS/PG stuff
   */

}

