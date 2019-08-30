#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <time.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>

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
   * Read command line options to get location of mnist data
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
    char c = getopt_long(argc, argv, "", long_options, &opt_idx);
    if(!c){
      if(!strcmp(long_options[opt_idx].name, "model"))   model_path  = optarg;
      if(!strcmp(long_options[opt_idx].name, "weights")) weight_path = optarg;
      if(!strcmp(long_options[opt_idx].name, "env"))     environment_name = optarg;
      args_read++;
    }else if(c == -1) break;
  }

  if(args_read < 3){
    if(!model_path)
      printf("Missing arg: --model [.sk file]\n");
    if(!weight_path)
      printf("Missing arg: --weights [.bin file]\n");
    if(!environment_name)
      printf("Missing arg: --env [envname]\n");
    exit(1);
  }

  Network n = sk_create_network(model_path);

  /*
   * TODO: Environment stuff
   * TODO: State normalization (part of model or part of env?)
   * TODO: ARS/PG stuff
   */

}
