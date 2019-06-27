#include <sieknet.h>

const char* modelfile = "model/test.sk";

int main(){
  Network n = create_network(modelfile);
  return 0;
}
