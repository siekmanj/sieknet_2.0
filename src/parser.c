#include <stdio.h>
#include <ctype.h>

#include <parser.h>
#include <sieknet.h>

#define BUFFSIZE 2048

static const char *sk_logistics[]           = {"sigmoid", "tanh", "relu", "softmax"};
static const char *sk_layertypes[]          = {"feedforward", "recurrent", "lstm", "attention"};
static const char *sk_layer_identifiers[]   = {"logistic", "size", "type", "input"};
static const char *sk_network_identifiers[] = {"input_dimension", "input", "output"};
static const char *sk_root_identifiers[]    = {"network", "layer"};

typedef enum sk_token_type {NETWORK_ROOT, LAYER_ROOT, COLON, COMMA, IDENTIFIER} TokenType;

typedef union value_{
  int number; 
  char* name; 
} Value;

typedef struct token_{
  Value value; 
  TokenType type; 
} Token;

// tokenize
// strip whitespace (might not be needed)
// build structs
// make sure no nonexistent layer names are used
// assign layer pointers
static int is_type(const char *value, const char **keywords, size_t len){
  for(int i = 0; i < len; i++){
    if(!strcmp(value, keywords[i])){
      return i; 
    }
  }
  return -1;  
}

static TokenType get_token_type(const char *type){
  //int idx = is_type(type, sk_token_types, sizeof(sk_token_types)/sizeof(char*));
  //printf("got %d for '%s'\n", idx, type);
  //return idx;
}


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

}

Layer *layer_from_string(const char *object){

}

Network network_from_string(const char *object){

}

Layer* parse_layer_attribute(Layer *l, const char *line){

}

Network* parse_network_attribute(Network *n, const char *line){

}

Network create_network(const char *skfile){
  if(!skfile)
    sk_err("null pointer");

  FILE *fp = fopen(skfile, "rb");

  if(!fp)
    sk_err("could not open file.");

  Network n;

  char *input_layers[5];
  char *output_layers[5];

  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);

  char *src = string_from_file(skfile);
  strip_string(src);
  size_t num_layers = layers_in_cfg(src);
  
  char *tmp = src;
  while(sscanf(tmp, "%s\n", buff) != EOF){
    
    Token t;

    printf("'%s'\n", buff);
    
    getchar();

    tmp += sizeof(char) * strlen(buff) + 1;
    memset(buff, '\0', BUFFSIZE);
  }

  free(src);

  return n;
}
