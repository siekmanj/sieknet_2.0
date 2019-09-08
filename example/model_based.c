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
    n = sk_create_network(model_path);

  Environment e;
  //if(!strcmp(environment_name, "humanoid"))
  //  e = create_humanoid_env();
  if(!strcmp(environment_name, "cassie"))
    e = create_cassie_env();

  Layer *out = sk_layer_from_name(&n, "action");
  if(!out)
    SK_ERROR("Whoops\n");

  if(out->size != e.action_space)
    SK_ERROR("Must match\n");

  if(n.input_dimension != e.observation_space)
    SK_ERROR("Must match! %lu vs %lu\n", n.input_dimension, e.observation_space);

  Tensor x = create_tensor(SIEKNET_CPU, n.input_dimension);
  e.reset(e);
  e.seed(e);
  for(size_t i = 0; i < 1e6; i++){
    Tensor current_out = get_subtensor(out->output, n.t);
    for(int j = 0; j < n.input_dimension; j++)
      tensor_raw(x)[tensor_get_offset(x, j)] = e.state[j];

    sk_forward(&n, x);

    float buff[out->size];
    for(int j = 0; j < out->size; j++)
      buff[j] = tensor_at(current_out, j);
    
    e.step(e, buff);
    e.render(e);
    if(*e.done){
      e.reset(e);
      e.seed(e);
    }
  }
  /*
   * TODO: Environment stuff
   * TODO: State normalization (part of model or part of env?)
   * TODO: ARS/PG stuff
   */

}
