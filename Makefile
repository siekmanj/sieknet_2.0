CC=gcc

SRC = src/*.c src/*/*.c
INC = -I src -I src/layers -I src/math -I src/parser -I src/env -I src/algo
LIB = -lm 

$(shell mkdir -p bin)

test:
	$(CC) -Wall example/test.c $(SRC) $(INC) $(LIB) -o bin/test

model_based:
	$(CC) -Wall example/model_based.c $(SRC) $(INC) $(LIB) -o bin/model_based

gradient_check:
	$(CC) -Wall example/gradient_check.c $(SRC) $(INC) $(LIB) -o bin/gradient_check

mnist:
	$(CC) -Wall example/mnist.c $(SRC) $(INC) $(LIB) -o bin/mnist

mj_demo:
	$(CC) -Wall example/mj_demo.c $(SRC) $(INC) $(LIB) -Lcgym -Icgym/include -lcgym
