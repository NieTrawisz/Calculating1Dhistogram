#include "readTextFile.h"
#include <stdio.h>
#include <cuda.h>

#define N_LETTERS 26

void seqentialHistogram(unsigned char *data, int length, unsigned int *histo, int nBins)
{
	for (int i = 0; i < length; i++)
	{
		int alphabetPosition = data[i] - 'a';
		if (alphabetPosition >= 0 && alphabetPosition < N_LETTERS)
		{
			histo[(alphabetPosition*nBins) / N_LETTERS]++;
		}
	}
}

// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long threadSize = ((size - 1) / (gridDim.x * blockDim.x)) + 1;

	long startPos = index * threadSize;
	for (long i = startPos; i < startPos + threadSize && i < size; i++)
	{
		int pos = (int)buffer[i];
		pos -= 'a';
		if (pos > -1 && pos < N_LETTERS)
			atomicAdd(&(histogram[(pos*nBins) / N_LETTERS]), 1);
	}
}

// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	long index = blockIdx.x * blockDim.x + threadIdx.x;
	long threadsCount = (gridDim.x * blockDim.x);

	if (index < size)
		for (long i = index; i < size; i += threadsCount)
		{
			int pos = (int)buffer[i];
			pos -= 'a';
			if (pos > -1 && pos < N_LETTERS)
				atomicAdd(&(histogram[(pos*nBins) / N_LETTERS]), 1);
		}
}

// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	extern __shared__ unsigned int s_histogram[];

	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
		s_histogram[i] = 0;

	__syncthreads();

	long blockSize = ((size - 1) / gridDim.x) + 1;
	long blockStart = blockIdx.x * blockSize;
	for (int i = blockStart + threadIdx.x; i < blockStart + blockSize && i < size; i += blockDim.x)
	{
		register int pos = (int)buffer[i];
		pos -= 'a';
		if (pos > -1 && pos < N_LETTERS)
			atomicAdd(&(s_histogram[(pos*nBins) / N_LETTERS]), 1);
	}

	__syncthreads();

	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
		atomicAdd(&(histogram[i]), s_histogram[i]);
}

// Extra: Histogram - interleaved partitioning + privatisation + aggregation
__global__ void histogram_4(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	extern __shared__ unsigned int s_histogram[];

	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
		s_histogram[i] = 0;

	__syncthreads();

	long blockSize = ((size - 1) / gridDim.x) + 1;
	long blockStart = blockIdx.x * blockSize;

	register int same_letters_counter = 0;
	int last_letter = -9999;
	for (int i = blockStart + threadIdx.x; i < blockStart + blockSize && i < size; i += blockDim.x)
	{
		register int pos = (int)buffer[i];
		pos -= 'a';
		if (pos == last_letter)
			same_letters_counter++;
		else
		{
			if (last_letter > -1 && last_letter < N_LETTERS)
				atomicAdd(&(s_histogram[(last_letter * nBins) / N_LETTERS]), same_letters_counter);
			
			same_letters_counter=1;
			last_letter=pos;
		}
	}

	if (last_letter > -1 && last_letter < N_LETTERS)
		atomicAdd(&(s_histogram[(last_letter * nBins) / N_LETTERS]), same_letters_counter);

	__syncthreads();

	for (int i = threadIdx.x; i < nBins; i += blockDim.x)
		atomicAdd(&(histogram[i]), s_histogram[i]);
}

int main(int argc, char **argv)
{
	// check if number of input args is correct: input text filename
	if (argc < 2 || argc > 3)
	{
		printf("Wrong number of arguments! Expecting 1 mandatory argument (input .txt filename) and 1 optional argument (number of bins). \n");
		return 0;
	}

	// read input string
	long size = getNoChars(argv[1]) + 1;
	unsigned char *h_buffer = (unsigned char *)malloc(size * sizeof(unsigned char));
	readFile(argv[1], size, h_buffer);
	printf("Input string size: %ld\n", size);

	// set number of bins
	int nBins = 26;
	if (argc == 3)
	{
		int inBinsVal = atoi(argv[2]);
		if (inBinsVal <= N_LETTERS)
		{
			nBins = inBinsVal;
		}
	}

	// histograms init
	unsigned int *histogram1 = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	unsigned int *histogram2 = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	unsigned int *histogram3 = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	unsigned int *histogram4 = (unsigned int *)malloc(nBins * sizeof(unsigned int));

	// cuda alloc
	unsigned char *d_buffer;
	unsigned int *distogram1;
	unsigned int *distogram2;
	unsigned int *distogram3;
	unsigned int *distogram4;

	cudaMalloc((void **)&d_buffer, size * sizeof(unsigned char));
	cudaMalloc((void **)&distogram1, nBins * sizeof(unsigned int));
	cudaMalloc((void **)&distogram2, nBins * sizeof(unsigned int));
	cudaMalloc((void **)&distogram3, nBins * sizeof(unsigned int));
	cudaMalloc((void **)&distogram4, nBins * sizeof(unsigned int));

	// cuda run
	cudaMemcpy(d_buffer, h_buffer, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 dimGrid1d(60);
	dim3 dimBlock1d(256, 1, 1);

	histogram_1<<<dimGrid1d, dimBlock1d>>>(d_buffer, size, distogram1, nBins);
	histogram_2<<<dimGrid1d, dimBlock1d>>>(d_buffer, size, distogram2, nBins);
	histogram_3<<<dimGrid1d, dimBlock1d, nBins>>>(d_buffer, size, distogram3, nBins);
	histogram_4<<<dimGrid1d, dimBlock1d, nBins>>>(d_buffer, size, distogram4, nBins);

	cudaMemcpy(histogram1, distogram1, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(histogram2, distogram2, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(histogram3, distogram3, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(histogram4, distogram4, nBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	// sequential
	unsigned int *histogram = (unsigned int *)malloc(nBins * sizeof(unsigned int));
	for (int i = 0; i < nBins; i++)
		histogram[i] = 0;

	seqentialHistogram(h_buffer, size, histogram, nBins);

	// printing
	printf("Histogram 0: ");
	for (int i = 0; i < nBins; i++)
		printf("%d ", histogram[i]);
	printf("\n");

	printf("Histogram 1: ");
	for (int i = 0; i < nBins; i++)
		printf("%d ", histogram1[i]);
	printf("\n");

	printf("Histogram 2: ");
	for (int i = 0; i < nBins; i++)
		printf("%d ", histogram2[i]);
	printf("\n");

	printf("Histogram 3: ");
	for (int i = 0; i < nBins; i++)
		printf("%d ", histogram3[i]);
	printf("\n");

	printf("Histogram 4: ");
	for (int i = 0; i < nBins; i++)
		printf("%d ", histogram4[i]);
	printf("\n");

	// free memory
	free(histogram);
	free(histogram1);
	free(histogram2);
	free(histogram3);
	free(histogram4);

	cudaFree(d_buffer);
	cudaFree(distogram1);
	cudaFree(distogram2);
	cudaFree(distogram3);
	cudaFree(distogram4);

	///////////////////////////////////////////////////////

	free(h_buffer);

	// For error detection you can use the following code (don't forget to include iostream)
	// cudaError_t err = cudaGetLastError();
	// if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

	return 0;
}
