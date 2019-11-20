default_target: test

.PHONY: clean test

main: main.cu
	nvcc -O3 -o main main.cu

clean:
	rm -f main

test: main
	srun -N1 -n1 -p all -w dnode02 ./main
