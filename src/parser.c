#include <stdio.h>
#include <ctype.h>

#include <parser.h>
#include <sieknet.h>

#define BUFFSIZE 2048

static const char *sk_logistics[]           = {"sigmoid", "tanh", "relu", "linear", "softmax"};
static const char *sk_layertypes[]          = {"feedforward", "recurrent", "lstm", "attention"};
static const char *sk_layer_identifiers[]   = {"logistic", "size", "type", "input", "name"};
static const char *sk_network_identifiers[] = {"input_dimension", "name", "input", "output"};
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
  if(c > 90 && c < 97 && c != 95)
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

static TokenType get_token_type(const char *token){
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

static void parse_layer_attribute(Layer *l, char *identifier, char **remaining){
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  int offset;

  if(sscanf(*remaining, "%s%n", buff, &offset) != EOF){

    char *original = *remaining;
    *remaining = *remaining + offset;

    if(!strcmp(identifier, "input")){
      if(l->input_names)
        SK_ERROR("a layer can only have one input layer(s) field.");

      size_t num_layers = 0;

      while(1){
        // Count the number of input layers
        if(get_token_type(buff) == VALUE){
          num_layers++;

          if(!(sscanf(*remaining, "%s%n", buff, &offset) != EOF))
            SK_ERROR("unexpected EOF while reading file.");
          *remaining += offset;
        }
        else
          break;
      }
      *remaining = original;
      l->input_names = calloc(num_layers, sizeof(char*));
      l->num_input_layers = num_layers;
      for(int i = 0; i < num_layers; i++){
        if((sscanf(*remaining, "%s%n", buff, &offset) != EOF)){
          *remaining += offset;

          char *name = calloc(strlen(buff), sizeof(char));
          strcpy(name, buff);
          l->input_names[i] = name;
        }
        else
          SK_ERROR("unexpected end of field while reading layer input names");
      }
    }
    else if(!strcmp(identifier, "logistic")){
      if(l->logistic != -1)
        SK_ERROR("a layer can have only one logistic function field, but found logistic functions '%d' (%s) and '%s'.", l->logistic, sk_logistics[l->logistic], buff);

      int logistic_idx = contains(buff, sk_logistics, STATIC_LEN(sk_logistics)); 
      if(logistic_idx != -1)
        l->logistic = logistic_idx;
      else
        SK_ERROR("could not find logistic function '%s'", buff);
    }
    else if(!strcmp(identifier, "type")){
      if(l->layertype != -1)
        SK_ERROR("a layer can only have one type field, but found types '%d' (%s) and '%s'.", l->layertype, sk_layertypes[l->layertype], buff);

      int layertype_idx = contains(buff, sk_layertypes, STATIC_LEN(sk_layertypes));
      if(layertype_idx != -1){
        l->layertype = layertype_idx;
      }else
        SK_ERROR("could not find layer type '%s'", buff);
    }
    else if(!strcmp(identifier, "size")){
      if(l->size)
        SK_ERROR("a layer can have only one size field, but found sizes '%lu' and '%s'.", l->size, buff);

      int size;
      if(sscanf(buff, "%d%n", &size, &offset)){
        l->size = size;
        if(l->size <= 0)
          SK_ERROR("layer size must be strictly greater than zero (got %lu).", l->size);
      }else{
        SK_ERROR("unable to parse layer size from '%s'", buff);
      }
    }
    else if(!strcmp(identifier, "name")){
      if(l->name)
        SK_ERROR("a layer can only have one name field, but found names '%s' and '%s'", l->name, buff);

      char *name = calloc(strlen(buff), sizeof(char));
      strcpy(name, buff);
      l->name = name;
    }
    else
      SK_ERROR("unrecognized identifier");
  }else
    SK_ERROR("unexpected EOF while reading file.");
}

static void parse_network_attribute(Network *n, char *identifier, char **remaining){
  char buff[BUFFSIZE];
  memset(buff, '\0', BUFFSIZE);
  int offset;

  if(sscanf(*remaining, "%s%n", buff, &offset) != EOF){
    *remaining = *remaining + offset;
    if(!strcmp(identifier, "name")){
      if(n->name)
        SK_ERROR("network can only have one name field.");

      char *name = calloc(strlen(buff), sizeof(char));
      strcpy(name, buff);
      n->name = name;
    }
    else if(!strcmp(identifier, "input_dimension")){
      if(n->input_dimension)
        SK_ERROR("only one input dimension field is allowed but found '%lu' and '%s'.", n->input_dimension, buff);

      int size;
      if(sscanf(buff, "%d%n", &size, &offset)){
        n->input_dimension = size;
      }else{
        SK_ERROR("unable to parse layer size");
      }
    }
    else if(!strcmp(identifier, "input")){
      if(n->input_layername)
        SK_ERROR("only one input layer field is allowed but found '%s' and '%s'.", n->input_layername, buff);

      char *name = calloc(strlen(buff), sizeof(char));
      strcpy(name, buff);
      n->input_layername= name;
    }
    else if(!strcmp(identifier, "output")){
      if(n->output_layername)
        SK_ERROR("only one output layer field is allowed but found '%s' and '%s'.", n->output_layername, buff);

      char *name = calloc(strlen(buff), sizeof(char));
      strcpy(name, buff);
      n->output_layername = name;
    }
    else
      SK_ERROR("unknown identifier: '%s'", identifier);
  }
  else
    SK_ERROR("unexpected EOF while parsing identifier '%s'.", identifier);
}

Network parse_network(const char *skfile){
  if(!skfile)
    SK_ERROR("null pointer");

  FILE *fp = fopen(skfile, "rb");

  if(!fp)
    SK_ERROR("could not open file.");

  Network n = {0};

  char *src = string_from_file(skfile);
  strip_string(src);
  size_t num_layers = layers_in_cfg(src);

  Layer **layers = malloc(sizeof(Layer*) * num_layers);
  for(int i = 0; i < num_layers; i++){
    layers[i] = calloc(sizeof(Layer), 1);
    layers[i]->logistic = -1;
    layers[i]->layertype = -1;
    layers[i]->size = 0;
  }

  int layeridx = -1;
  
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
    else if(current_token == LAYER_ROOT){

      current_root = current_token;
      layeridx++;
    
    }
    
    else if(current_token == IDENTIFIER){

      if(current_root == NETWORK_ROOT)
        parse_network_attribute(&n, buff, &tmp);

      else if(current_root == LAYER_ROOT)
        parse_layer_attribute(layers[layeridx], buff, &tmp);

    }
    memset(buff, '\0', BUFFSIZE);
  }

  n.layers = layers;
  n.depth = num_layers;

  for(int i = 0; i < n.depth-1; i++){
    for(int j = i + 1; j < n.depth; j++){
      const char *name    = n.layers[i]->name;
      const char *compare = n.layers[j]->name;

      if(!name || !compare)
        SK_ERROR("every layer object must have a name attribute.");
      
      if(!strcmp(name, compare))
        SK_ERROR("cannot have duplicate layer names (layer %d, '%s', and layer %d, '%s'", i, name, j, compare);
    }
  }
  for(int i = 0; i < n.depth; i++){
    if(n.layers[i]->logistic == -1)
      SK_ERROR("'%s' must have a logistic function attribute.", n.layers[i]->name);
    
    if(n.layers[i]->layertype == -1)
      SK_ERROR("'%s' must have a layertype attribute.", n.layers[i]->name);

    for(int j = 0; j < (int)n.layers[i]->num_input_layers - 1; j++){
      for(int k = j + 1; k < n.layers[i]->num_input_layers; k++){
        const char *name    = n.layers[i]->input_names[j];
        const char *compare = n.layers[i]->input_names[k];
        if(!strcmp(name, compare))
          SK_ERROR("'%s' has duplicate input layers '%s' and '%s'.", n.layers[i]->name, name, compare);

      }
    }
  }
  
  if(!n.name)
    SK_ERROR("could not find 'name' attribute for network.");

  free(src);
  return n;
}


