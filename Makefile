CC=gcc

SRC = src/*.c src/*/*.c
INC = -I src -I src/layers -I src/math -I src/parser -I src/env -I src/algo
LIB = -lm

CGYM_ROOT = $(HOME)/cgym
CGYM_INC = -I$(CGYM_ROOT) -I$(CGYM_ROOT)/include -L$(CGYM_ROOT)
CGYM_LIB = -lmjenvs -lcassieenvs

$(shell mkdir -p bin)

test:
	$(CC) -Wall example/test.c $(SRC) $(INC) $(LIB) -o bin/test

gradient_check:
	$(CC) -Wall example/gradient_check.c $(SRC) $(INC) $(LIB) -o bin/gradient_check

mnist:
	$(CC) -Wall example/mnist.c $(SRC) $(INC) $(LIB) -o bin/mnist

ddpg:
	$(CC) -Wall example/ddpg_train.c $(SRC) $(INC) $(CGYM_INC) $(LIB) $(CGYM_LIB) -o bin/ddpg

ars:
	$(CC) -Wall example/random_search.c $(SRC) $(INC) $(CGYM_INC) $(LIB) $(CGYM_LIB) -fopenmp -DSIEKNET_USE_OMP -o bin/ars

eval_policy:
	$(CC) -Wall example/eval_policy.c $(SRC) $(INC) $(CGYM_INC) $(LIB) $(CGYM_LIB) -o bin/eval_policy -DCOMPILED_FOR_MUJOCO
