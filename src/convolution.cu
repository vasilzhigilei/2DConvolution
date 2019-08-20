//============================================================================
// 2D Convolution
// A CUDA 2D convolution implementation for GPGPUs
//
// Main file for testing and running the CUDA kernel
//
// Written by:    Vasil Zhigilei
//============================================================================

// includes, system
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// texture input memory
texture<int, 2, cudaReadModeElementType> texIn;

// constant filter memory
__constant__ int filter[9];

// number of threads per block
const int THREADS_PER_BLOCK = 256;

__global__ void convolution1(int *output, int DIM){
	/*
	 * 2D convolution using constant filter memory and texture input memory optimizations
	 */
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// check if thread within DIM bounds
	if(x < DIM && y < DIM){
		// initialize dot product
		int dot_product = 0;
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				// compute dot product with input data and filter
				dot_product += tex2D(texIn, x + i - 1, y + j - 1) * filter[i + j*3];
			}
		}
		// copy dot product to device output memory
		output[offset] = dot_product;
	}
}

int main(){
	/*
	 * Main function for running the 2D convolution kernel
	 */

	// dimension of one side of the 2D input data
	int DIM = 1024;
	// memory size required for input
	int SIZE = DIM * DIM * sizeof(int);

	// seed generation for random, use same seed for same results every run
	srand(10);

	// host memory allocation
	int *h_input = (int *) malloc(SIZE);
	int *h_output = (int *) malloc(SIZE);
	int *h_filter = (int *) malloc(9*sizeof(int));

	// device memory allocation
	int *d_input, *d_output;
	cudaMalloc((void**) &d_input, SIZE);
	cudaMalloc((void**) &d_output, SIZE);

	// texture memory binding to input
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	cudaBindTexture2D(0, &texIn, d_input, &desc, DIM, DIM, DIM*sizeof(int));

	// generate input data
	cout << "Generating input data..." << endl;
	for(int i = 0; i<DIM; i++){
		for(int j = 0; j<DIM; j++){
			h_input[i+j*DIM] = rand() % 11 - 5;
		}
	}

	// copy input memory to device
	cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice);

	// Generate filter
	cout << "Generating filter..." << endl;
	for(int i = 0; i<3; i++){
		for(int j = 0; j<3; j++){
			h_filter[i+j*3] = rand() % 11 - 5;
		}
	}

	// copy filter to constant device memory
	cudaMemcpyToSymbol(filter, h_filter, 9*sizeof(int));

	// using THREADS_PER_BLOCK, calculate grid and block size
	dim3 grid_size(16*((DIM+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK), 16*((DIM+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK), 1);
	dim3 block_size(THREADS_PER_BLOCK/16, THREADS_PER_BLOCK/16, 1);

	// run convolution kernel
	convolution1<<<grid_size, block_size>>>(d_output, DIM);

	// copy results back to host
	cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost);

	// console output some of the outputs
	for(int i = 0; i<DIM*DIM-1; i+=DIM*DIM/10){
		cout << i << ": " << h_output[i] << endl;
	}



	// clean up memory on host & device
	free(h_input);
	free(h_output);
	free(h_filter);

	cudaUnbindTexture(texIn);

	cudaFree(d_input);
	cudaFree(d_output);

}
