#include<stdio.h>

/*MACROS Here
*/

/*GLOBAL VARIABLES HERE
*/


class Genome{
	private:
		int sizeOfGenome;
		float  fitnessValue;
		float (*fitness)(Genome& g,char* dna);
	public:
		char *genome;
		Genome();
		Genome(int sg,float (*fit)(Genome& g,char* dna));
		Genome(Genome &g);
		~Genome();
		void initGenome();
		void calFitness(char* s);
		float getFitnessValue();
		int getSize();
		void setFitnessValue(float fitness);
		 
};




class EABase : public Genome
{
	private:
		Genome *population;
		Genome type;
		int sizeofgenome;
		int noOfGenerations;
		int populationSize;
		float pmutation;
		float score;
		void crossover(Genome, Genome&);
		void initFitness(char* s);
		
	public: 
		EABase(Genome& g);   		// EA Base Constructor
		~EABase();					// EA base Destructor
		int getPopulationSize();
		void setPopulationSize(int);
		int getSizeOfGenome();
		void setSizeOfGenome(int);
		void initializePopulation();
		void mutation(Genome&);
		void setGenerations(int);
		int getGenerations();
		void setMutationProbability(float);
		float getGenerationScore();
		void evolve(char* s);
		void doCrossover(int start, int end);
		void sortPopulation();
		float getMinFitness();
		
};

