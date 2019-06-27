#include <stdio.h>
#include <ctype.h>

#include <parser.h>
#include <sieknet.h>

#define BUFFSIZE 2048

static const char *sk_logistics[]   = {"sigmoid", "tanh", "relu", "softmax"};
static const char *sk_identifiers[] = {"network", "layer", "name", "logistic", "size"};

static const char sk_delimiters[] = {'{', '}', ':', ','};

typedef enum sk_token_type {NETWORK_ROOT, LAYER_ROOT, LAYER_NAME, LOGISTIC_FUNCTION, LAYER_TYPE, LAYER_SIZE, BRACKET, COLON, COMMA, IDENTIFIER} TokenType;

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

TokenType get_expected_token(const char *key){
   
}

bool is_type(const char *value, const char **keywords, size_t len){
  for(int i = 0; i < sizeof(keywords) / sizeof(char*); ) 
}

char *string_from_file(const char *filename){
  FILE *fp = fopen(filename, "rb");
  
  fseek(fp, 0, SEEK_END);
  size_t bytes = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *ret = calloc(bytes, sizeof(char));
  
  if(ret)
    fread(ret, 1, bytes, fp);
  
  fclose(fp);
  return ret;
}

void strip_whitespace(char *str){
}

void str_to_lower(char *str){
  for(int i = 0; i < strlen(str); i++)
    str[i] = tolower(str[i]);
}

Layer *layer_from_string(const char *object){

}

Network network_from_string(const char *object){

}

Layer* parse_layer(){

}

Network create_network(const char *skfile){
  if(!skfile)
    sk_err("null pointer");

  FILE *fp = fopen(skfile, "rb");

  if(!fp)
    sk_err("could not open file.");

  Network n;

  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  
  while(fscanf(fp, "%s", buff) != EOF){
    str_to_lower(buff);
    Token t;
    printf("'%s'\n", buff);
    getchar();


    memset(buff, '\0', BUFFSIZE);
  }
  return n;
}
