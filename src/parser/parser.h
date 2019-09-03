#ifndef SIEKNET_PARSER_H
#define SIEKNET_PARSER_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sieknet.h>

int sk_parser_find_int(const char *, char *, int *);

int sk_parser_find_string(const char *, char *, char **);

int sk_parser_find_strings(const char *, char *, char ***, size_t *);

int sk_parser_get_line(char **, char *, size_t *);

char *sk_parser_string_from_file(const char *filename);
#endif
