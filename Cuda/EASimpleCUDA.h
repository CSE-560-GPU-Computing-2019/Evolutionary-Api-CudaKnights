class Genome{
	private:
		int sizeOfGenome;
		float  fitnessValue;
		float (*fitness)(Genome& g,char* dna);
	public:
		char *genome;
		__device__ __host__ Genome();
		__device__ __host__ Genome(int sg,float (*fit)(Genome& g,char* dna));
		__device__ __host__ Genome(Genome &g);
		__device__ __host__ ~Genome();
		 void initGenome();
		  void calFitness(char* s);
		 float getFitnessValue();
		 __device__ __host__ int getSize();
		 void setFitnessValue(float fitness);
		 
};




class EABase 
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
		__device__ __host__ EABase(Genome& g);   		// EA Base Constructor
		__device__ __host__ ~EABase();					// EA base Destructor
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

