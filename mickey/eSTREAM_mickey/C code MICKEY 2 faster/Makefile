all: d_mickey test
d_mickey: mickey_kernel.c
	gcc -o d_mickey mickey_kernel.c
test: mickey2.c test_kg.c ecrypt-sync.[ch] xsr_rng.o
	gcc -o test xsr_rng.o mickey2.c test_kg.c  
xsr_rng.o: ../../../xsr_rng.[ch]
	gcc -c ../../../xsr_rng.c
