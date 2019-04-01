#include "EASimple.h"
#include <stdlib.h> 
#include <algorithm>


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
	#pragma omp parallel for  default(none) shared(genome,g,sizeOfGenome)
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
	#pragma omp parallel for default(none) shared(population, populationSize)
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
	float min = 99999999999.0;
	for(int i=0;i<populationSize;i++){
		if(min>population[i].getFitnessValue()){
			min = population[i].getFitnessValue();
		}
	}
	return min;
}
void EABase::crossover(Genome &genome1,Genome &genome2)
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
	#pragma omp parallel for shared(population) firstprivate(start,end)
	for(int i=start;i<end-1;i+=2){
		crossover(population[i],population[i+1]);
		mutation(population[i]);
		mutation(population[i+1]);
	}
}
void EABase::evolve(char* s)
{	
	initializePopulation();
	float currentFitnessValue=0.0;
	int threshold_count=20;
	float percentageOfTop  =0.1;
	int count=0;
	for(int i=0;i<noOfGenerations*5;i++){
		initFitness(s);
		sortPopulation();
		int start = (int) 0.1 * populationSize;
		int end = populationSize-1;
		Shuffle(population,start,end);
		float fit = population[0].getFitnessValue();
		if(fit==currentFitnessValue){
			count++;
		} 
		else{
			count=0;
		}
		if(count>=threshold_count){
			break;
		}

		doCrossover(start,end);

	}

}

void EABase::sortPopulation(){
	quickSort(population,0,populationSize-1);
}
void EABase::initFitness(char* s){
	#pragma omp parallel for default(none) shared(s,population,populationSize)
	for(int i=0;i<populationSize;i++){
		population[i].calFitness(s);
	}
}
