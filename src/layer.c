#include <string.h>

#include <layer.h>

Layer *layer_from_name(Network *n, const char *name){
  for(int i = 0; i < n->depth; i++){
    if(!strcmp(n->layers[i]->name, name))
      return n->layers[i];
  }
  return NULL;
}

int contains_layer(Layer **arr, Layer *comp, size_t arrlen){
  for(int i = 0; i < arrlen; i++){
    if(arr[i] == comp)
      return 1;
  }
  return 0;
}

