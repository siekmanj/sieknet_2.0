CC=gcc

SRC = src/*.c src/*/*.c main.c
INC = -I include -I include/layers -I include/math -I include/parser
LIB = -lm 

default:
	$(CC) $(SRC) $(INC) $(LIB) -Wall -O3 -std=c99
