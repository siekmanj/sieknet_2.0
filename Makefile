CC=gcc

SRC = src/*.c src/*/*.c
INC = -I include -I include/layers -I include/math -I include/parser
LIB = -lm 

default:
	$(CC) -Wall main.c $(SRC) $(INC) $(LIB)

mnist:
	$(CC) -Wall mnist.c $(SRC) $(INC) $(LIB)
