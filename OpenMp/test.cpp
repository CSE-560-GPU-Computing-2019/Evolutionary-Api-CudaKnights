#include <stdlib.h> 
#include "EASimple.h"
#include <iostream>
#include<chrono>

using namespace std::chrono;
using namespace std;

char* initialize(int len)
{
	char* dna = new char[len];
	for (int i = 0; i < len; i++)
	{
		int s = rand()%10;
		if(s<3)
			dna[i] = 'A';
		else if (s < 5)
			dna[i] = 'T';
		else if (s < 8)
			dna[i] = 'G';
		else
			dna[i] = 'C';
	}
	return dna;
}

float fitnessFunction(Genome& g, char*s)
{
	float f = 0;
	int size = g.getSize();
	for (int i = 0; i < size; i++)
		if(g.genome[i] != s[i])
			f++;
	return f;
}

int main(int argc, char* argv[])
{

	// cout<<"Initialized";
	srand(time(0));
	char* s = initialize(100000);
	Genome g(100000,fitnessFunction);
	g.initGenome();
	EABase ea(g);
	ea.setPopulationSize(100);
	auto start = high_resolution_clock::now();
	ea.evolve(s);
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout<<"time = "<<duration.count()/1000000.0<<endl;
	cout<<"Min fitness: "<<ea.getMinFitness()<<endl;
	// cout<<fitnessFunction(g,s)<<endl;
	// g.calFitness(s);
	// cout<<endl;
	// cout<<g.getFitnessValue()<<endl;


}