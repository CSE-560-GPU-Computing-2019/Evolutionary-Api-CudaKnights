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

using namespace std;


float *valuesc;
float *weightc;
float maxWc;

char *matchc;
 const int cSUMFLAG=0;
 const int cKNAPSACKFLAG = 1;

const int cAVGFLAG=2;
const int cMATCHFLAG=3;
const int cINVERSESUMFLAG=4;

 const int cMAXIMIZE=-1;
 const int cMINIMIZE=1;



__global__ void setup_kernelc ( curandState *state, unsigned long seed )
{
    curand_init ( seed, 0, 0, &state[0] );
} 

__device__ float generateRandomc( curandState* globalState) 
{
    //int ind = threadIdx.x;
    curandState localState = globalState[0];
    float RANDOM = curand_uniform( &localState );
    globalState[0] = localState;
    return RANDOM;
}



__global__ void initializeChar1Dpopulation(char *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division){
	int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
	if(populationIndex<(sizeofPopulation*sizeofChormosome)){
	population[populationIndex]= (char) ((int) (generateRandomc(globalState)*2)+48);
	//printf("CUDA %d\n",population[populationIndex]);
	
	}
__syncthreads();
}


__global__ void initializeChar1DRangepopulation(char *population,int sizeofPopulation,int sizeofChormosome,curandState* globalState,int division,char* range){
	int populationIndex =  blockIdx.x * blockDim.x + threadIdx.x;
	if(populationIndex<(sizeofPopulation*sizeofChormosome)){
	population[populationIndex]= range[(int) (generateRandomc(globalState)*sizeofChormosome)];
	//printf("CUDA %d\n",population[populationIndex]);
	
	}
__syncthreads();
}


__device__ void Crossover(char *chromosome,int size,int start1,int end1,int start2,int end2,int  mid){

	for(int i=mid;i<size;i++){
		int c1 = start1+mid;
		int c2 = start2+mid;
		int temp = chromosome[c1];//
			//printf("temp =%d and c1 = %d and c2 = %d and ch[c2] = %d\n",temp,c1,c2,chromosome[c2]);
			
			chromosome[c1]=	chromosome[c2];
			chromosome[c2]=temp;
	}	
}

void EAChar::shuffle(int bias){

	std::random_shuffle(population+bias, population+populationSize);
	
}

__device__ float calculateFitnessBinary(char *chromosome,int flag,int size,float *value,float *weight,float maxLimit,char *match,int start, int end){
 	switch(flag){
 		case cKNAPSACKFLAG:
 			return getKnapsackFitnessc(chromosome,size,value,weight,maxLimit,start,end);
 		case cSUMFLAG:
 			return getSumc(chromosome,size,start,end);
 		case cAVGFLAG:
 			return getSumc(chromosome,size,start,end)/size;
 		case cMATCHFLAG:
 			return getMatchc(chromosome,match,size,start,end);
 		default: return 0.0;
 	}
 
}

__device__ float getKnapsackFitnessc(char *chromosome, int size,float* values,float *weight,float maxW, int start,int end){
	float totalWeight=0.0;
	float value =0.0;	
	for(int i=0;i<size&&start+i<end;i++){
		float w = (chromosome[start+i]-48)*weight[i];
		float v = (chromosome[start+i]-48)*values[i];
		if(w+totalWeight<=maxW){
			value+=v;
			totalWeight+=w;
		}

	}
	return value;
}
__global__ void gpuCrossover(char *chromosome,curandState *globalState,int sizeofChromosome,int sizeofPopulation,int Bias,float prob,char* range){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int mid =(int) (generateRandomc(globalState)*sizeofChromosome);//4;// (int) (generateRandom(globalState)*(sizeofChromosome-1));
	//printf("MID: %d\n", mid);
	idx=idx*2;
	int start1,end1;

	int start2,end2;
	start1 = idx*sizeofChromosome;
	end1 = start1+sizeofChromosome;
	start2 = end1;
	end2 = start2+sizeofChromosome;
	if(end2<(sizeofChromosome*sizeofPopulation) )
	Crossover(chromosome,sizeofChromosome,start1,end1,start2,end2,mid);
	int number = (int) (generateRandomc(globalState)*100);
	if(number<(prob*100)){
		int j = (int) (generateRandomc(globalState)*((int)sizeofChromosome/4));
		for(int k=0;k<j;k++){
			int index = (int) (generateRandomc(globalState)*sizeofChromosome);
			chromosome[index] = range[(int) (generateRandomc(globalState)*sizeofChromosome)];// = //(chromosome[index]+1)%2;
			// if(a=='1'){
			// 	chromosome[index]='0';

			// } 
			// else{
			// 	chromosome[index]='1';				
			// }
		}
	} 

}
__global__ void calculateFitness(char *chromosome,int fitnessFlag,float *fitnessValues,int sizeofPopulation,int sizeofChromosome,float *value,float *weight,float maxLimit,char *match){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int start,end;
	start =idx*sizeofChromosome;
	end = (idx+1)*sizeofChromosome;
	fitnessValues[idx] = calculateFitnessBinary(chromosome,fitnessFlag,sizeofChromosome,value,weight,maxLimit,match,start,end);

}
void EAChar::init()
{

//printf("Hello %d and %d\n",chromosomeSize,populationSize );
			srand(time(0));
			//curandState* devStates;
			auto start = chrono::steady_clock::now();
		    cudaMalloc ( &devStates, sizeof( curandState ) );
		    auto end = chrono::steady_clock::now();
		    
		    double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
		    totalMemoryTransferTime+=elapsed_seconds;

		    start = chrono::steady_clock::now();		    
		    setup_kernelc <<< 1, 1>>> ( devStates,unsigned(time(NULL)) );
			end = chrono::steady_clock::now();	
	
			elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
		    totalKernelTime+=elapsed_seconds;
			
			threads = dim3(chromosomeSize,1);
			blocks = dim3(populationSize,1);
			start = chrono::steady_clock::now();		
			
		   	initializeChar1DRangepopulation<<<blocks,threads>>>(Cudapopulation,chromosomeSize,popSize,devStates,4,randomRange);
			end = chrono::steady_clock::now();		    
		   	elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
		   	totalKernelTime+=elapsed_seconds;
			cudaDeviceSynchronize();
  
			start = chrono::steady_clock::now();		    
		   	
			for(int i=0;i<populationSize;i++){

			cudaMemcpy(population[i].chromosome, Cudapopulation+(i*chromosomeSize), sizeof(char)*chromosomeSize, cudaMemcpyDeviceToHost);
			//printf("Initialized: %s \t size: %d\n", population[i].chromosome,chromosomeSize);
			
			}
			end = chrono::steady_clock::now();		    
		   	
			elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
			totalMemoryTransferTime+=elapsed_seconds;

			start = chrono::steady_clock::now();		    
		   	
			fitness(0);
			end = start = chrono::steady_clock::now();		    
		   elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
			fitnessCalculationTime+=elapsed_seconds;
}

void EAChar::setFitnessFlag(int fit,int minimax){
	fitnessFlag = fit;
	minmaxflag = minimax;
}

void EAChar::doCrossOver(int bias){
	int s = populationSize/2;\
	/*CharChromosome1D *pop1;
	CharChromosome1D *pop2;
	
	pop1 = new CharChromosome1D[populationSize];
	pop2 = new CharChromosome1D[populationSize];

	for(int i=0;i<populationSize;i++){
	 	pop1[i].initializeChromosome(chromosomeSize);
	 	pop1[i].fitnessValue = population[i].fitnessValue;
	 	pop2[i].initializeChromosome(chromosomeSize);
	 }*/

	auto start = chrono::steady_clock::now();		    
	
	for(int i=0;i<populationSize;i++){
			cudaMemcpy( Cudapopulation+(i*chromosomeSize),population[i].chromosome, sizeof(char)*chromosomeSize, cudaMemcpyHostToDevice);
			//cudaMemcpy( pop1[i].chromosome,Cudapopulation+(i*chromosomeSize), chromosomeSize*sizeof(int), cudaMemcpyDeviceToHost);
			
	 		
	}
	auto end = chrono::steady_clock::now();		    
	
	double elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	
	totalMemoryTransferTime+=elapsed_seconds;
	bias=0;


	shuffle(bias);
	if(populationSize<256){
		threads=dim3(s,1);
		blocks=dim3(1,1);
	}
	else{
	threads = dim3(256,1);
	blocks = dim3(ceil(s/256),1);
	}

	start = chrono::steady_clock::now();		    
	
	gpuCrossover<<<blocks,threads>>>(Cudapopulation,devStates,chromosomeSize,populationSize,bias,mutationProbability,randomRange);

	end = chrono::steady_clock::now();		    
	elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	totalKernelTime +=elapsed_seconds;


	start = chrono::steady_clock::now();		    
	
	for(int i=0;i<populationSize;i++){
			cudaMemcpy(population[i+populationSize].chromosome, Cudapopulation+(i*chromosomeSize), sizeof(char)*chromosomeSize, cudaMemcpyDeviceToHost);
		//	cudaMemcpy(pop2[i].chromosome, Cudapopulation+(i*chromosomeSize), chromosomeSize*sizeof(int), cudaMemcpyHostToDevice);
	}
	end = chrono::steady_clock::now();		    
	
	elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	
	totalMemoryTransferTime+=elapsed_seconds;


	start = chrono::steady_clock::now();		    
		   	
	fitness(populationSize);
	end = start = chrono::steady_clock::now();		    
	elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	fitnessCalculationTime+=elapsed_seconds;
	
	start = chrono::steady_clock::now();
	sortpop();
	end = chrono::steady_clock::now();
	elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	sortingpopulationTime+=elapsed_seconds;	
/*
	int l=0,m=0;
for(int i = 0; i < populationSize; ++i) {
			
						pop2[i].chromosome = population[i].chromosome;		
						pop2[i].fitnessValue = population[i].fitnessValue;
		}	
		int i=0;
		while(i<populationSize){
			if(pop2[l].fitnessValue<=pop1[m].fitnessValue){
				l++;
				i++;
			}
			else{
				for(int s=0;s<chromosomeSize;s++)	
				population[i].chromosome[s] = pop1[m].chromosome[s];

				i++;
				m++;
			}
		}
		delete pop1;
		delete pop2;
*/

}
void EAChar::doMutation(int bias){


	for(int i=bias;i<populationSize;i++){
		int j=0;
		int number = rand()%100;
		
		if(number<mutationProbability*100){
			 j = rand()%chromosomeSize;
			 for(int k=0;k<j;k++){
			 	int index = rand()%chromosomeSize;

			 	//printf("MUTATING %d\n",index );
			 	population[i].chromosome[index] = (char) (((int)population[i].chromosome[index]+1-48)%2 + 48);  
			 }
		}
	}
}
EAChar::EAChar(int sofc,int sofp,char *range)
{
	populationSize = sofp;
	popSize= populationSize*2;
	chromosomeSize = sofc;

	population = new CharChromosome1D[popSize];
	//cudaMalloc(&Cudapopulation1D, populationSize);
	
	for(int i=0;i<popSize;i++){
	 	population[i].initializeChromosome(chromosomeSize);
	 	//cudaMalloc((void **)&Cudapopulation1D[i], chromosomeSize);
	 	//indices[i] =(rand() % chromosomeSize-1) + 2;
	 }


	 //cudaMalloc((void **)&cudaIndices,populationSize*sizeof(int));
	 //cudaMemcpy(cudaIndices, indices, populationSize*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Cudapopulation, populationSize*chromosomeSize*sizeof(char));
	cudaMalloc((void**)&randomRange, chromosomeSize*sizeof(char));
	cudaMemcpy(randomRange,range, chromosomeSize*sizeof(char), cudaMemcpyHostToDevice);
	
}
void EAChar::printpopulation()
{
	
	//sortpop();
	for(int i=0;i<2*populationSize;i++){
		for(int j=0;j<chromosomeSize;j++){
			printf("%c",population[i].chromosome[j]);

		}
		printf("\t %f\n",population[i].fitnessValue*minmaxflag);
	}
}
float EAChar::fitness(int tag){
	float *fitnessV;
	float *realfitness;
	float avgFitness=0.0;
	realfitness = (float*) malloc(popSize*sizeof(float));
	cudaMalloc((void**)&fitnessV, popSize*sizeof(float));
	if(populationSize<256){
		threads=dim3(populationSize,1);
		blocks=dim3(1,1);

	}
	else{
	threads = dim3(256,1);
	blocks = dim3(ceil(popSize/256),1);
	}
	auto start = chrono::steady_clock::now();
	calculateFitness<<<blocks,threads>>>(Cudapopulation,fitnessFlag,fitnessV,popSize,chromosomeSize,valuesc,weightc,maxWc,matchc);
	auto end = chrono::steady_clock::now();
	double elapsed_seconds =  std::chrono::duration_cast<std::chrono::duration<double> >(end-start).count();
	totalKernelTime+=elapsed_seconds;

	cudaMemcpy(realfitness, fitnessV,popSize*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0;i<popSize;i++){
		population[i+tag].fitnessValue=realfitness[i]*minmaxflag;
		avgFitness+=realfitness[i];
	}
	avgFitness/=popSize;
	//printf("Average Fitness: %f\n",avgFitness);
	return avgFitness;
}

void EAChar::sortpop(){
 std::sort(population, population + popSize,[](CharChromosome1D const & a, CharChromosome1D const & b) -> bool 
 			{ return (a.fitnessValue) < (b.fitnessValue); } );
}
void EAChar::evolve()
{	int bias = 0.2 * populationSize;
	for(int i=0;i<100;i++){
		float a= fitness(0);
		if (i == 99)
			printf("Avg fitness:  %f\n", a);
		sortpop();
		doCrossOver(0);
		//printpopulation();
		//doMutation(bias);
	}

}

void EAChar::setParamKnapSack(float *kvalues,float *kweight,int chromosomeSize,float maxWeight){
	cudaMalloc((void **)&valuesc,chromosomeSize*sizeof(float));
	cudaMalloc((void **)&weightc,chromosomeSize*sizeof(float));
	maxWc = maxWeight;
	cudaMemcpy(valuesc,kvalues,chromosomeSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(weightc,kweight,chromosomeSize*sizeof(float), cudaMemcpyHostToDevice);
}
void EAChar::setMatchParameter(char *kvalues,int chromosomeSize){
	cudaMalloc((void **)&matchc,chromosomeSize*sizeof(char));
	cudaMemcpy(matchc,kvalues,chromosomeSize*sizeof(char), cudaMemcpyHostToDevice);

}

__device__ float getMatchc(char *chromosome,char *match,int size,int start,int end)
{
	float c = 0;
	for(int i = start,j=0; i < end&&j<size; i++,j++)
		if (chromosome[i] != match[j])
			c += 1;
	return c;
}

__device__ float getSumc(char *chromosome,int size,int start,int end)
{
	float c = 0;
	for(int i = start; i < end; i++)
		c += (int)chromosome[i]-48;
	return c;
}

/*=======================================================================================================*/