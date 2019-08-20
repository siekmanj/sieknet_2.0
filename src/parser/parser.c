#include <stdio.h>
#include <ctype.h>

#include <parser.h>
#include <sieknet.h>
#include <layer.h>

#define BUFFSIZE 2048

static char *string_from_file(const char *filename){
  FILE *fp = fopen(filename, "rb");
  
  fseek(fp, 0, SEEK_END);
  size_t bytes = ftell(fp) + 1;
  fseek(fp, 0, SEEK_SET);

  char *ret = calloc(bytes, sizeof(char));
  
  if(ret)
    if(!fread(ret, 1, bytes, fp))
      SK_ERROR("Unable to read from '%s'.", filename);
  
  fclose(fp);
  return ret;
}

int sk_parser_get_line(char **str, char *buf, size_t *len){
  char *start = *str;
  if(!str || !*str || **str == '\0')
    return 0;

  while(**str != '\n' && **str != '\0') (*str)++;
  size_t bytes = *str - start;

  if(buf){
    memcpy(buf, start, bytes);
    buf[bytes] = '\0';
  }

  if(**str != '\0')
    (*str)++;

  if(len)
    *len = bytes;
  return 1;
}

static size_t layers_in_cfg(char *str){
  char buff[BUFFSIZE];
  size_t len;
  size_t num_layers = 0;
  while(sk_parser_get_line(&str, buff, &len)){
    if(sk_layer_parse_identifier(buff) != -1)
      num_layers++;
  }
  return num_layers;
}


static int is_filler(char c){
  if(c < 48 && (c != '\n' && c != ' '))
    return 1;
  if(c > 57 && c < 65)
    return 1;
  if(c > 90 && c < 97 && c != '_' && c != '[' && c != ']')
    return 1;
  if(c > 123)
    return 1;
  return 0;
}

void sk_parser_strip_string(char *str){
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

int sk_parser_find_int(const char *identifier, char *src, int *dest){
  char line[BUFFSIZE];
  while(sk_parser_get_line(&src, line, NULL)){
    sk_parser_strip_string(line);
    char arg1[BUFFSIZE];
    int arg2 = 0;
    if(sscanf(line, "%s %d\n", arg1, &arg2) == 2){
      if(!strcmp(arg1, identifier)){
        *dest = arg2;
        return 1;
      }
    }
  }
  return 0;
}

int sk_parser_find_string(const char *identifier, char *src, char **dest){
  char line[BUFFSIZE];
  while(sk_parser_get_line(&src, line, NULL)){
    sk_parser_strip_string(line);
    char arg1[BUFFSIZE];
    char arg2[BUFFSIZE];
    if(sscanf(line, "%s %s\n", arg1, arg2) == 2){
      if(!strcmp(arg1, identifier)){
        *dest = (char*)malloc((strlen(arg2)+1)*sizeof(char));
        strcpy(*dest, arg2);
        return 1;
      }
    }
  }
  return 0;
}

int sk_parser_find_strings(const char *identifier, char *src, char ***dest, size_t *num){
  char line[BUFFSIZE];
  while(sk_parser_get_line(&src, line, NULL)){
    sk_parser_strip_string(line);
    char arg1[BUFFSIZE];
    if(sscanf(line, "%s", arg1) == 1){
      if(!strcmp(arg1, identifier)){
        size_t offset = strlen(arg1) + 1;

        size_t num_args = 0;
        char argn[BUFFSIZE];
        while(offset < strlen(line)){
          if(sscanf(line + sizeof(char) * offset, "%s", argn) == 1)
            num_args++;
          offset += strlen(argn)+1;
        }

        size_t counter = 0;
        *dest = (char**)malloc(num_args * sizeof(char*));
        offset = strlen(arg1) + 1;
        while(offset < strlen(line)){
          if(sscanf(line + sizeof(char) * offset, "%s", argn) == 1){
            (*dest)[counter] = (char*)malloc((strlen(argn)+1) * sizeof(char));
            strcpy((*dest)[counter], argn);
            counter++;
          }
          offset += strlen(argn)+1;
        }

        *num = num_args;
      }
    }
  }
  return 0;
}

#if 0

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

#endif

#if 0
static void parse_layer_attribute(Layer *l, char *identifier, char **remaining){
#if 0
  sk_parse_layer_attribute(l, identifier, remaining);
#else
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
#endif
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
#endif

void parse_network(Network *n, const char *skfile){
  if(!skfile)
    SK_ERROR("null pointer");

  FILE *fp = fopen(skfile, "rb");

  if(!fp)
    SK_ERROR("could not open file '%s'", skfile);

  char *src = string_from_file(skfile);
  size_t num_layers = layers_in_cfg(src);

  Layer **layers = malloc(sizeof(Layer*) * num_layers);
  for(int i = 0; i < num_layers; i++){
    layers[i] = calloc(sizeof(Layer), 1);
    layers[i]->logistic = -1;
    layers[i]->layertype = -1;
    layers[i]->weight_initialization = 0;
    layers[i]->size = 0;
  }

  char *tmp = src;
  char *start = src;
  char *end = src;
  char line[BUFFSIZE] = {0};
  int done = 0;
  for(int i = 0; i < num_layers; i++){
    /* Search for a start token */
    do {
      start = tmp;
      done = !sk_parser_get_line(&tmp, line, NULL);
    }
    while(sk_layer_parse_identifier(line) == -1 && !done);

    do {
      end = tmp;
      done = !sk_parser_get_line(&tmp, line, NULL);
    }
    while(sk_layer_parse_identifier(line) == -1 && strcmp(line, "[network]") && !done);

    char layer_src[end - start + 1];
    memcpy(layer_src, start, end - start);
    layer_src[end - start] = '\0';

    sk_layer_parse(layers[i], layer_src);

    tmp = end;

  }

  n->layers = layers;
  n->depth = num_layers;

  tmp = src;

  do {
    start = tmp;
    done = !sk_parser_get_line(&tmp, line, NULL);
  }
  while(strcmp("[network]", line) && !done);

  do {
    done = !sk_parser_get_line(&tmp, line, NULL);
    end = tmp;
  }
  while(sk_layer_parse_identifier(line) != 1 && !done);
  char network_src[end - start + 1];
  memcpy(network_src, start, end - start);
  network_src[end - start] = '\0';

  if(!sk_parser_find_string("input", network_src, &n->input_layername))
    SK_ERROR("No matching attribute 'input' found in [network] section.");
  
  if(!sk_parser_find_string("output", network_src, &n->output_layername))
    SK_ERROR("No matching attribute 'output' found in [network] section.");

  if(!sk_parser_find_string("name", network_src, &n->name))
    SK_ERROR("No matching attribute 'name' found in [network] section.");

  int indim = 0;
  if(!sk_parser_find_int("input_dimension", network_src, &indim))
    SK_ERROR("No matching attribute 'input_dimension' found in [network] section.");
  n->input_dimension = indim;

  free(src);
}


