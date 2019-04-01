all: 
	g++ -std=c++11 -c EASimple.cpp EASimple.h -fopenmp
	g++ -std=c++11 -o test test.cpp EASimple.o -fopenmp