CC=gcc

SRC = src/*.c main.c
INC = -I include
LIB = -lm 

default:
	$(CC) $(SRC) $(INC) $(LIB) -Wall -O3
