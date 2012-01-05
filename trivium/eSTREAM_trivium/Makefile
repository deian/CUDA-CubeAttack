
CFLAGS=-Wall -O3 -DECRYPT_API="ecrypt-sync.h"

test: test.o ecrypt-sync.o trivium.o d_trivium.o

test.o: ecrypt-sync.h d_trivium.h d_trivium.o
ecrypt-sync.o: ecrypt-sync.h
trivium.o: ecrypt-sync.h
dtrivium.o: d_trivium.c d_trivium.h


