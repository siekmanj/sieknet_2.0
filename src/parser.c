#include <stdio.h>
#include <ctype.h>

#include <parser.h>
#include <sieknet.h>

#define BUFFSIZE 2048

static const char *sk_logistics[]           = {"sigmoid", "tanh", "relu", "linear", "softmax"};
static const char *sk_layertypes[]          = {"feedforward", "recurrent", "lstm", "attention"};
static const char *sk_layer_identifiers[]   = {"logistic", "size", "type", "input", "name"};
static const char *sk_network_identifiers[] = {"inputdimension", "input", "output"};
static const char *sk_root_identifiers[]    = {"network", "layer"};

typedef enum sk_token_type {NETWORK_ROOT, LAYER_ROOT, IDENTIFIER, VALUE} TokenType;

static char *string_from_file(const char *filename){
  FILE *fp = fopen(filename, "rb");
  
  fseek(fp, 0, SEEK_END);
  size_t bytes = ftell(fp) + 1;
  fseek(fp, 0, SEEK_SET);

  char *ret = calloc(bytes, sizeof(char));
  
  if(ret)
    fread(ret, 1, bytes, fp);
  
  fclose(fp);
  return ret;
}

static int is_filler(char c){
  if(c < 48 && (c != 10 && c != 32))
    return 1;
  if(c > 57 && c < 65)
    return 1;
  if(c > 90 && c  < 97)
    return 1;
  if(c > 123)
    return 1;
  return 0;
}

static int contains(const char *str, const char **arr, size_t len){
  for(int i = 0; i < len; i++){
    if(!strcmp(str, arr[i]))
      return i;
  }
  return -1;
}

static int is_identifier(const char *token){
  if(contains(token, sk_layer_identifiers, STATIC_LEN(sk_layer_identifiers)) != -1)
    return 1;
  if(contains(token, sk_network_identifiers, STATIC_LEN(sk_network_identifiers)) != -1)
    return 1;
  return 0;
}

TokenType get_token_type(const char *token){
  if(!strcmp(token, "network"))
    return NETWORK_ROOT;
  if(!strcmp(token, "layer"))
    return LAYER_ROOT;
  if(is_identifier(token))
    return IDENTIFIER;
  return VALUE;
}

static void strip_string(char *str){
  size_t len = strlen(str) + 1;
  char *tmp = (char*)calloc(len, sizeof(char));
  size_t idx = 0;
  for(int i = 0; i < strlen(str); i++)
    if(!is_filler(str[i]))
      tmp[idx++] = tolower(str[i]);
  
  memset(str, '\0', strlen(str)+1);
  memcpy(str, tmp, strlen(tmp)+1);
  free(tmp);
}

static size_t layers_in_cfg(const char *str){
  size_t num_layers = 0;
  const char *tmp = str;
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  while(sscanf(tmp, "%s", buff) != EOF){
    if(!strcmp(buff, "layer"))
      num_layers++;
    
    tmp += sizeof(char) * strlen(buff) + 1;
    memset(buff, '\0', BUFFSIZE);
  }
  return num_layers;
}

Layer *layer_from_string(const char *object){

}

Network network_from_string(const char *object){

}

void parse_layer_attribute(Layer *l, char *identifier, char **remaining){
  printf("  PARSING LAYER ATTR '%s'\n", identifier);
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  int offset;

  if(sscanf(*remaining, "%s%n", buff, &offset) != EOF){

    *remaining = *remaining + offset;
    if(!strcmp(identifier, "input")){
      while(1){
        if(get_token_type(buff) == VALUE){

          offset = 0;
          memset(buff, '\0', BUFFSIZE);
          if(!(sscanf(*remaining, "%s%n", buff, &offset) != EOF))
            SK_ERROR("unexpected EOF while reading file.");
          *remaining = *remaining + offset;
        }
        else
          return;
      }
    }
    else if(!strcmp(identifier, "logistic")){
      int logistic_idx = contains(buff, sk_logistics, STATIC_LEN(sk_logistics)); 
      if(logistic_idx != -1){

      }else
        SK_ERROR("invalid logistic function");
    }
    else if(!strcmp(identifier, "type")){
      int layertype_idx = contains(buff, sk_layertypes, STATIC_LEN(sk_layertypes));
      if(layertype_idx != -1){

      }else
        SK_ERROR("invalid layer type");
    }
    else if(!strcmp(identifier, "size")){
      int size;
      if(sscanf(buff, "%d%n", &size, &offset)){
        printf("  got size '%d'\n", size);
      }
    }
    else if(!strcmp(identifier, "name")){
      printf("  got layername '%s'\n", buff);
    }
    else
      SK_ERROR("unrecognized identifier");
  }else
    SK_ERROR("unexpected EOF while reading file.");
}

void parse_network_attribute(Network *n, char *identifier, char **remaining){

}

Network create_network(const char *skfile){
  if(!skfile)
    SK_ERROR("null pointer");

  FILE *fp = fopen(skfile, "rb");

  if(!fp)
    SK_ERROR("could not open file.");

  Network n = {0};



  char *src = string_from_file(skfile);
  strip_string(src);
  size_t num_layers = layers_in_cfg(src);

  printf("config has %lu layers!\n", num_layers);

  Layer **layers = malloc(sizeof(Layer*) * num_layers);
  for(int i = 0; i < num_layers; i++){
    layers[i] = calloc(sizeof(Layer), 1);
    layers[i]->input_names = calloc(num_layers, sizeof(char*));
  }

  size_t layeridx = 0;
  
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  char *tmp = src;

  int seen_network_root = 0;
  TokenType current_root;
  TokenType current_token;

  int offset;
  while(sscanf(tmp, "%s%n", buff, &offset) != EOF){
    TokenType current_token = get_token_type(buff);

    tmp += offset;

    if(current_token == NETWORK_ROOT){
      current_root = current_token;

      if(seen_network_root)
        SK_ERROR("found more than one 'network' identifier - only one is allowed in a single file.");
      else
        seen_network_root = 1;
    }
    else if(current_token == LAYER_ROOT)
      current_root = current_token;
    
    else if(current_token == IDENTIFIER){

      if(current_root == NETWORK_ROOT)
        parse_network_attribute(&n, buff, &tmp);

      else if(current_root == LAYER_ROOT)
        parse_layer_attribute(layers[layeridx++], buff, &tmp);
    }
    //else
    //  SK_ERROR("expected a network root marker, layer root marker, or identifier but got a value.");

    //printf("remaining:\n'%s'\n", tmp);
    //getchar();
    memset(buff, '\0', BUFFSIZE);

  }

  free(src);

  return n;
}
