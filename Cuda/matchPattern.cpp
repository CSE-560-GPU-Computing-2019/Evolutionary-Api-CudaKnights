#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <algorithm>  // For time()
#include <cstdlib>
#include <chrono>
#include <unistd.h>
using namespace std;
float fitnessValue(char *chromosome,int size, char *match){
	float count=0.0;
	for(int i=0;i<size;i++){
		if(chromosome[i]!=match[i]){
			count +=1.0;
		}

	}
	return count;
}

class Population
{
	public:
		char *chromosome;
		int size;
		float fitness;
		void setChromosome(char *chr,int sizec){
			chromosome =(char *)malloc(sizeof(char)*sizec);
			for(int i=0;i<sizec;i++){
				chromosome[i]=chr[i];
			}
			size = sizec;
		}
		void getFitness(char *match){
			fitness = fitnessValue(chromosome,size,match);
		}

};
void initializePopulation(Population *p,int sizec){
	char c[sizec];
	for(int i=0;i<sizec;i++){
		int cx =rand()%2;
		if(cx==0){
			c[i]='0';

		}
		else{
			c[i]='1';
		}
	}
	p->setChromosome(c,sizec);

}
void crossover(Population *p1,Population *p2,Population *p3,Population *p4,int sizec){
	int mid  =  rand()%sizec;
	mid/=2;
	for(int i=0;i<sizec;i++){
		if(i>=mid){
			p3->chromosome[i]=p2->chromosome[i];
			p4->chromosome[i]=p1->chromosome[i];
			
		}
		else{
			p3->chromosome[i]=p1->chromosome[i];
			p4->chromosome[i]=p2->chromosome[i];
	
		}

	}
}
void mutation(Population *p, int mutationProb,int sizec){
	float r = rand();
	if(r<mutationProb){
		int times = rand()%(sizec/4);
		for(int i=0;i<times;i++){
			int index = rand()%sizec;
			char cc = p->chromosome[index];
			if(cc =='1'){
				 p->chromosome[index]='0';
			}
				else{
				 p->chromosome[index]='1';
				}
		}
	}
}
// void EABinary::shuffle(int bias){

// 	std::random_shuffle(population+bias, population+populationSize);
	
// }

void printchormosome(Population *p){
	int size = p->size;
	for(int i=0;i<size;i++){
		std::cout<<p->chromosome[i]<<" ";
	}
	std::cout<<" \n";
}

int main(int argc, char** argv){
	srand(time(0));
	auto start = chrono::steady_clock::now();
	int sizec = 1000;
	int sizep = 256;
	char c[sizec];
	for(int i=0;i<sizec;i++){
		int cx =rand()%2;
		if(cx==0){
			c[i]='0';

		}
		else{
			c[i]='1';
		}
	}
	Population p[2*sizep];
	//p[0].setChromosome(c,3);

	//printf("%c\n",p[0].chromosome[0]);
	for(int i=0;i<2*sizep;i++)
	{
		initializePopulation(&p[i],sizec);
	}
	for(int i=0;i<2*sizep;i++){
		p[i].getFitness(c);
	}
	//std::cout<<(char)rand()%2;
	   std::sort(p, p+(2*sizep),[](Population const & a,Population const & b) -> bool 
  	 		{ return (a.fitness) < (b.fitness); } );
	for(int i=0;i<100;i++){
		 for(int i=0;i<2*sizep;i++){
			p[i].getFitness(c);
		}
		 std::sort(p, p+(2*sizep),[](Population const & a,Population const & b) -> bool 
  			 { return (a.fitness) < (b.fitness); } );
		   std::random_shuffle(p, p+sizep);
			
		  for(int pop=0;pop<sizep;pop+=2){
		  	crossover(&p[pop],&p[pop+1],&p[pop+sizep],&p[pop+sizep+1],sizec);
		  	crossover(&p[pop],&p[pop+1],&p[pop+sizep],&p[pop+sizep+1],sizec);
		  }

		  for(int pop=sizep;pop<2*sizep;pop++){
		  	mutation(&p[pop],0.4,sizec);
		  }

		  for(int i=0;i<2*sizep;i++){
			p[i].getFitness(c);
		}
	
			   std::sort(p, p+(2*sizep),[](Population const & a,Population const & b) -> bool 
  			 { return (a.fitness) < (b.fitness); } );
		float avg=0.0;
			for (int ii = 0; ii < sizep; ++ii)
			{
				avg += p[ii].fitness;
				//printchormosome(&p[ii]);
			}
		avg/=sizep;

		printf("Average fitness = %f\n",avg);

	}
	auto end = chrono::steady_clock::now();
	double elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	std::cout<<"Time: "<<elapsed_seconds;
}