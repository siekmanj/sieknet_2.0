CC=gcc

SRC = src/*.c src/*/*.c main.c
INC = -I include -I include/layers -I include/math -I include/parser
LIB = -lm 

default:
	$(CC) -Wall $(SRC) $(INC) $(LIB)
