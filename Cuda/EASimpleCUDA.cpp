#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>
#include "EASimpleCUDA.h"

#include <cuda.h>
#include <curand_kernel.h>

__global__ void doCrossoverMutation(int start, int end, Genome *population,Genome *newPopulation,int length,int randIndex){
	int index = blockIdx.x*blockDim.x+threadIdx.x+start;
	//int randIndex = rand()%length;
	if( index<end)
	{
		for(int i=randIndex;i<length;i++){
		// char temp = genome1.genome[i];
		// genome1.genome[i] = genome2.genome[i];
		// genome2.genome[i] = temp;
	
		char temp = population[index].genome[i];
		population[index].genome[i] = population[index+1].genome[i];
		population[index+1].genome[i] = temp;
		
		}

	}
	newPopulation[index] = population[index];
	__syncthreads();
}
void printPop(Genome arr[],int sizeOfGenome,int size){
	for(int i=0;i<size;i++){
		for (int j = 0; j < sizeOfGenome; ++j)
		{
			//printf("%c ",arr[i].genome[j]);
		}
		printf("%f\n",arr[i].getFitnessValue());
	}printf("\n");
}

void swap(Genome* a, Genome* b) 
{ 
    Genome t = *a; 
    *a = *b; 
    *b = t; 
} 
char getCharacter(){
	int s = rand()%4;
		if(s<1)
			return 'A';
		else if (s < 2)
			return 'T';
		else if (s < 3)
			return 'G';
		else
			return 'C';
		
}
int partition (Genome *arr, int low, int high) 
{ 
    float pivot = arr[high].getFitnessValue(); 
    int i = (low - 1); 
  
    for (int j = low; j <= high- 1; j++) 
    { 
  
        if (arr[j].getFitnessValue() <= pivot) 
        { 
            i++;  
            swap(&arr[i], &arr[j]); 
        } 
    } 
    swap(&arr[i + 1], &arr[high]); 
    return (i + 1); 
} 
void quickSort(Genome *arr, int low, int high) 
{ 
    if (low < high) 
    { 
        int pi = partition(arr, low, high); 
        quickSort(arr, low, pi - 1); 
        quickSort(arr, pi + 1, high); 
    } 
} 
void Shuffle(Genome arr[],int start, int end){
	for(int i =start;i<=end;i++){
		int index =(rand() % (end - i + 1)) + i;
		swap(&arr[i],&arr[index]); 
	}
}


Genome::Genome()
{
	sizeOfGenome = 100;
	genome = new char[100];
	fitnessValue = 0.0;
	fitness = NULL;
}


Genome::Genome(int sg,float (*fit)(Genome& g,char* dna) = NULL)
{
	fitness = fit;
	sizeOfGenome = sg;
	genome = new char[sg];
	fitnessValue = 0.0;
}

void Genome::initGenome()
{
	for (int i = 0; i < sizeOfGenome; i++)
	{
		int s = rand()%4;
		if(s<1)
			genome[i] = 'A';
		else if (s < 2)
			genome[i] = 'T';
		else if (s < 3)
			genome[i] = 'G';
		else
			genome[i] = 'C';
	}
	//initalize a genome with Randon value
}


Genome::Genome(Genome &g)
{
	sizeOfGenome = g.sizeOfGenome;
	fitnessValue = g.fitnessValue;
	fitness = g.fitness;
	genome = new char[sizeOfGenome];
	for(int i = 0; i < sizeOfGenome; i++)
		genome[i] = g.genome[i];
}

float Genome::getFitnessValue()
{
	return fitnessValue;
}

void Genome::calFitness(char* s)
{
	Genome *g = this;
	if(fitness != NULL)
		fitnessValue = fitness(*g,s);
}
void Genome::setFitnessValue(float fitness)
{
	fitnessValue = fitness;
}

int Genome::getSize(){
	return sizeOfGenome;
}

Genome::~Genome()
{

}

EABase::~EABase()
{
	
}

EABase::EABase(Genome& g)
{
	type = g;
	populationSize = 100;
	noOfGenerations = 10;
	pmutation = 0.001;
}


int EABase::getPopulationSize()
{
	return populationSize;
}


void EABase::setPopulationSize(int s)
{
	populationSize = s;
}

int EABase::getGenerations()
{
	return noOfGenerations;
}

void EABase::setGenerations(int s)
{
	noOfGenerations = s;
}

int EABase::getSizeOfGenome()
{
	return sizeofgenome;
}

void EABase::initializePopulation()
{
	population = new Genome[populationSize];
	for (int i = 0; i < populationSize; i++)
	{
		Genome g(type);
		g.initGenome();
		population[i] = g;
	}
}

float EABase::getGenerationScore()
{
	return score;
}

float EABase::getMinFitness(){
	float min = population[0].getFitnessValue();
	for(int i=0;i<populationSize;i++){
		if(min>population[i].getFitnessValue()){
			min = population[i].getFitnessValue();
		}
	}
	return min;
}
void EABase::crossover(Genome genome1,Genome &genome2)
{
	int length = genome1.getSize();
	int index = rand()%length;

	for(int i=index;i<length;i++){
		char temp = genome1.genome[i];
		genome1.genome[i] = genome2.genome[i];
		genome2.genome[i] = temp;
	}
}

void EABase::mutation(Genome &genome)
{
	int i1 = rand()%genome.getSize();
	int i2 = rand()%genome.getSize();
	genome.genome[i1] = getCharacter();

}
void EABase::doCrossover(int start, int end){
	Genome *newPopulation;
	Genome *currPopulation;
	int randIndex = rand()%populationSize;
	int sizeofpop = sizeof(Genome)*populationSize;
	cudaMalloc((void**)&newPopulation, sizeofpop);
	cudaMalloc((void**)&currPopulation, sizeofpop);
	cudaMemcpy(currPopulation, population, sizeofpop, cudaMemcpyHostToDevice);
	const dim3 blockSize(1024,1,1);
	const dim3 gridSize(((float) populationSize+1/1024),1);
	doCrossoverMutation<<<gridSize,blockSize>>>(start,end,currPopulation,newPopulation,population[0].getSize(),randIndex);
	cudaMemcpy(population, newPopulation, sizeofpop, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(newPopulation);
	cudaFree(currPopulation);
}
void EABase::evolve(char* s)
{	
	initializePopulation();
	float currentFitnessValue=0.0;
	int threshold_count=20;
	float percentageOfTop  =0.1;
	int count=0;
	for(int i=0;i<noOfGenerations*100;i++){
		initFitness(s);
		//printf("%f and noOfGenerations = %d \n",getMinFitness(),noOfGenerations);
		sortPopulation();
		printPop(population,population[0].getSize(),populationSize);
		int start = (int) 0.1 * populationSize+1;
		int end = populationSize-1;
		Shuffle(population,start,end);
		float fit = population[0].getFitnessValue();
		if(fit==currentFitnessValue){
			count++;
		} 
		else{
			currentFitnessValue = fit;
			count=0;
		}
		if(count>=threshold_count){
			//break;
		}

		doCrossover(start,end);
		
			}

}

void EABase::sortPopulation(){
	quickSort(population,0,populationSize-1);
}
void EABase::initFitness(char* s){
	for(int i=0;i<populationSize;i++){
		population[i].calFitness(s);
	}
}
