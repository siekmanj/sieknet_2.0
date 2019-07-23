#ifndef SIEKNET_CONF_H
#define SIEKNET_CONF_H

#ifndef SIEKNET_MAX_UNROLL_LENGTH
#define SIEKNET_MAX_UNROLL_LENGTH 100
#endif

#ifndef SIEKNET_DEBUG
#define SIEKNET_DEBUG
#endif


#define SK_ERROR(...) \
  do {                                                                                    \
    fprintf(stderr, "\nFATAL ERROR: ");                                                   \
    fprintf(stderr, "in file %s:%d - function '%s()'):\n", __FILE__, __LINE__, __func__); \
    fprintf(stderr, "             ");                                                     \
    fprintf(stderr, __VA_ARGS__);                                                         \
    fprintf(stderr, "             ");                                                     \
    fprintf(stderr, "\nExiting immediately.\n");                                          \
    exit(1);                                                                              \
  } while(0)

#define STATIC_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define MAX(a,b) ((float)a > (float)b ? a : b)
#define MIN(a,b) ((float)a < (float)b ? a : b)
#define SWAP(a,b) \
  do {            \
    a ^= b;       \
    b ^= a;       \
    a ^= b;       \
  } while(0)

#endif

