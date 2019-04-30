#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <ctime>
#include <algorithm>  // For time()
#include <cstdlib>
#include <chrono>
#include <unistd.h>

#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#include "EASimpleChar.h"
#include "EASimpleBinary.h"
#include "EASimpleFloat.h"

using namespace std::chrono;


int main(int argc,char **argv){
	srand(time(0));
	int sizec = 100000;
	int sizep = 255;

	char arr[sizec];//={0,1,1,0,1,1,1,0,1,0};
	float c[sizec],v[sizec];
	for(int i=0;i<sizec;i++){
		int cx =rand()%50;
		int vx = rand()%30;
		c[i]=cx;
		v[i]=vx;
	}

	float m=100;

	for(int i=0;i<sizec;i++){
		arr[i]= (char)(rand()%2+48);
		//printf("%d\n",arr[i]);
	}
	auto start = high_resolution_clock::now();

	EAChar CHAR = EAChar(sizec,sizep,arr);
		printf("Char\n");
	//CHAR.setParamKnapSack(v,c,sizec,m);

	CHAR.setMatchParameter(arr,sizec);
	//CHAR.setFitnessFlag(KNAPSACKFLAG,MAXIMIZE);
	CHAR.setFitnessFlag(MATCHFLAG,MAXIMIZE);
	CHAR.initializePopulation();
	//BINARY.printpopulation();
	
	CHAR.evolve();
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout<<"Char time = "<<duration.count()/1000000.0<<std::endl;

	//BINARY.printpopulation();
	// std::cout<<"------------------------------CHARACTER-----------------------------"<<std::endl;
	// CHAR.getStatistics();



	int arrb[sizec];
	for(int i=0;i<sizec;i++){
		arrb[i]= (rand()%2);
		//printf("%d\n",arr[i]);
	}
	auto start1 = high_resolution_clock::now();

	EABinary BINARY = EABinary(sizec,sizep,arrb);
		printf("Int\n");
	BINARY.setParamKnapSack(v,c,sizec,m);

	//BINARY.setMatchParameter(arrb,sizec);
	BINARY.setFitnessFlag(KNAPSACKFLAG,MAXIMIZE);
	//BINARY.setFitnessFlag(MATCHFLAG,MINIMIZE);
	BINARY.initializePopulation();
	//BINARY.printpopulation();
	
	BINARY.evolve();
	//BINARY.printpopulation();
	auto stop1 = high_resolution_clock::now(); 
	auto duration1 = duration_cast<microseconds>(stop1 - start1);
	std::cout<<"Int time = "<<duration1.count()/1000000.0<<std::endl;

	float arrf[sizec];
	for(int i=0;i<sizec;i++){
		arrf[i]= (float)((int)(rand()%2));
		//printf("%d\n",arr[i]);
	}

	auto start2 = high_resolution_clock::now();


	EAFloat Float = EAFloat(sizec,sizep,arrf);
	printf("Float\n");
	Float.setParamKnapSack(v,c,sizec,m);

	//Float.setMatchParameter(arrf,sizec);
	Float.setFitnessFlag(KNAPSACKFLAG,MAXIMIZE);
	//Float.setFitnessFlag(MATCHFLAG,MINIMIZE);
	Float.initializePopulation();
	//BINARY.printpopulation();
	
	Float.evolve();
	//BINARY.printpopulation();

	auto stop2 = high_resolution_clock::now(); 
	auto duration2 = duration_cast<microseconds>(stop2 - start2);
	std::cout<<"Float time = "<<duration2.count()/1000000.0<<std::endl;


	std::cout<<"------------------------------CHARACTER-----------------------------"<<std::endl;
	CHAR.getStatistics();

	std::cout<<"------------------------------BINARY-----------------------------"<<std::endl;
	BINARY.getStatistics();

	std::cout<<"------------------------------Float-----------------------------"<<std::endl;
	Float.getStatistics();

	// printf("Match: ");
	// for(int i=0;i<sizec;i++){
	// 	//arr[i]= (char)(rand()%2+48);
	// 	printf("%c",arr[i]);
	// }

	// printf("\n");
	

	return 0;
}
