CC=gcc

SRC = src/*.c src/*/*.c
INC = -I include -I include/layers -I include/math -I include/parser
LIB = -lm 

default:
	$(CC) -Wall main.c $(SRC) $(INC) $(LIB)

model_based:
	$(CC) -Wall example/model_based.c $(SRC) $(INC) $(LIB) -o bin/model_based

gradient_check:
	$(CC) -Wall example/gradient_check.c $(SRC) $(INC) $(LIB) -o bin/gradient_check

mnist:
	$(CC) -Wall example/mnist.c $(SRC) $(INC) $(LIB) -o bin/mnist
