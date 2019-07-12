#ifndef SIEKNET_CONF_H
#define SIEKNET_CONF_H

#ifndef SIEKNET_MAX_UNROLL_LENGTH
#define SIEKNET_MAX_UNROLL_LENGTH 100
#endif


#define SK_ERROR(...) \
  do {                                                                                          \
    fprintf(stderr, "ERROR: ");                                                                 \
    fprintf(stderr, __VA_ARGS__);                                                               \
    fprintf(stderr, "\n (From file %s:%d - function '%s()')\n", __FILE__, __LINE__, __func__);  \
    exit(1);                                                                                    \
  } while(0)

#define STATIC_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define MAX(a,b) ((float)a > (float)b ? a : b)

#endif

