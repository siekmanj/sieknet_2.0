#include <sieknet.h>
#include <stdio.h>

const char* test0= "model/one.sk";
const char* test1= "model/two.sk";
const char* test2= "model/three.sk";

int main(){
  printf("one:\n");
  Network one = create_network(test0);
  /*
  printf("two:\n");
  Network two = create_network(test1);
  printf("three:\n");
  Network thr = create_network(test2);
  */
  return 0;
}
