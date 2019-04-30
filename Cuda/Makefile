all: 
	nvcc -std=c++11 -c EASimpleChar.cu
	nvcc -std=c++11 -c EASimpleBinary.cu
	nvcc -std=c++11 -c EASimpleFloat.cu
	nvcc -std=c++11 -o test test.cu EASimpleChar.o EASimpleBinary.o EASimpleFloat.o -g