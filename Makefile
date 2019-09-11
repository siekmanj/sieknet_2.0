CC=gcc

SRC = src/*.c src/*/*.c
INC = -I src -I src/layers -I src/math -I src/parser -I src/env -I src/algo
LIB = -lm

CGYM_INC = -Icgym/ -Icgym/include -Lcgym
CGYM_LIB = -lmjenvs -lcassieenvs

$(shell mkdir -p bin)

test:
	$(CC) -Wall example/test.c $(SRC) $(INC) $(LIB) -o bin/test

gradient_check:
	$(CC) -Wall example/gradient_check.c $(SRC) $(INC) $(LIB) -o bin/gradient_check

mnist:
	$(CC) -Wall example/mnist.c $(SRC) $(INC) $(LIB) -o bin/mnist

model_based:
	$(CC) -Wall example/model_based.c $(SRC) $(INC) $(CGYM_INC) $(LIB) $(CGYM_LIB) -o bin/model_based

random_search:
	$(CC) -Wall example/random_search.c $(SRC) $(INC) $(CGYM_INC) $(LIB) $(CGYM_LIB) -o bin/ars
