#include <stdio.h>
#include <stdlib.h>

#include <sieknet.h>
#include <parser.h>


Network create_network(const char *skfile){
  Network n = {0};

  /* 
   * Retrieve layer names + sizes, input layer names, 
   * logistic functions, layer types, and network name.
   */
  parse_network(&n, skfile);

  /*
   * Construct directed graph, assign input & output layers,
   * determine order of execution and recurrent inputs.
   */
  build_network(&n);

  /*
   * 
   *
   */

  return n;
}
