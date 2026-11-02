#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define MAX_MASK_WIDTH 10
#define TILE_SIZE 256
__constant__ float DM[MAX_MASK_WIDTH];


typedef struct {
	int width;
	int maskWidth;

	float *N;
	float *M;
	float *P;

	float *dN;
	float *dM;
	float *dP;
	float *dP2;
	float *P2;
	float *P3;

} Context;


void convolution1DCPU(Context *ctx) {

	for (int i = 0; i < ctx->width; i++) {

		float pValue = 0;
		int nStartPoint = i - (ctx->maskWidth / 2);
		for (int j = 0; j < ctx->maskWidth; j++) {
			if (nStartPoint + j >= 0 && nStartPoint + j < ctx->width) {
				pValue += ctx->N[nStartPoint + j] * ctx->M[j];
			}
		}
		ctx->P[i] = pValue;
	}
}

__global__ void convolution1DGPU(float *N, float *P,
	int maskWidth, int width) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float pValue = 0;
	int nStartPoint = idx - (maskWidth / 2);
	for (int j = 0; j < maskWidth; j++) {
		if (nStartPoint + j >= 0 && nStartPoint + j < width) {
			pValue += N[nStartPoint + j] * DM[j];
		}
	}

	P[idx] = pValue;
}

__global__ void tiledConvolution1D(float *N, float *P,
	int maskWidth, int width) {

	__shared__ float nDS[TILE_SIZE + MAX_MASK_WIDTH - 1];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int n = maskWidth / 2;

	int haloIndexLeft = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
	if (threadIdx.x >= blockDim.x - n) {
		nDS[threadIdx.x - (blockDim.x - n)] = (haloIndexLeft < 0) ? 0 : N[haloIndexLeft];
	}

	nDS[n + threadIdx.x] = N[(blockIdx.x * blockDim.x) + threadIdx.x];

	int haloIndexRight = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
	if (threadIdx.x < n) {
		nDS[n + blockDim.x + threadIdx.x] = (haloIndexRight >= width) ? 0 : N[haloIndexRight];
	}

	__syncthreads();

	float pValue = 0;
	for (int j = 0; j < maskWidth; j++) {
		pValue += nDS[j + threadIdx.x] * DM[j];
	}

	P[idx] = pValue;
}

void foo(Context *ctx) {

	// printf("Input Array\n");
	// for (int i = 0 ; i < ctx.width; ++i) printf("%f\n", ctx.N[i]);
	// printf("Mask Array\n");
	// for (int i = 0 ; i < ctx->maskWidth; ++i) printf("%f\n", ctx->M[i]);
	printf("Output(CPU) Array\n");
	for (int i = 0 ; i < ctx->width; ++i) printf("%f\n", ctx->P[i]);

	printf("Output(GPU) Array\n");
	for (int i = 0 ; i < ctx->width; ++i) printf("%f\n", ctx->P2[i]);

	// printf("dis(gen): %f\n", dis(gen));
}

void compareCPUToGPU(Context *ctx) {

	for (int i = 0; i < ctx->width; i++) {

		if (fabs(ctx->P2[i] - ctx->P[i]) >= 0.00001) {
			printf("Not correct!\n");
		}

		if (fabs(ctx->P3[i] - ctx->P[i]) >= 0.00001) {
			printf("Not correct!\n");
		}
	}
}

int main() {

	// Loading an image.
	int x, y, n;
	unsigned char * imgData = stbi_load("hoh-river-valley.jpg", &x, &y, &n, 0);
	


	Context ctx;
	ctx.width = 1 << 20;
	ctx.maskWidth = 7;

	ctx.N 	= 	(float *) malloc(ctx.width 		* sizeof(float));
	ctx.M 	= 	(float *) malloc(ctx.maskWidth 	* sizeof(float));
	ctx.P 	= 	(float *) malloc(ctx.width 		* sizeof(float));
	ctx.P2 	= 	(float *) malloc(ctx.width 		* sizeof(float));
	ctx.P3 	= 	(float *) malloc(ctx.width 		* sizeof(float));


	std::mt19937 gen(1);
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);


	for (int i = 0; i < ctx.width; ++i) ctx.N[i] = dis(gen);
	for (int i = 0; i < ctx.maskWidth;  ++i) ctx.M[i] = dis(gen);


	auto begin = std::chrono::high_resolution_clock::now();
	convolution1DCPU(&ctx);
	auto end = std::chrono::high_resolution_clock::now();
	
	auto elapsedCPU = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	std::cout << elapsedCPU.count() << " milliseconds" << std::endl;

	cudaMalloc(&ctx.dN, ctx.width * sizeof(float));
	cudaMemcpy(ctx.dN, ctx.N, ctx.width * sizeof(float), 
		cudaMemcpyHostToDevice);
	cudaMalloc(&ctx.dM, ctx.maskWidth * sizeof(float));
	cudaMemcpy(ctx.dM, ctx.M, ctx.maskWidth * sizeof(float), 
		cudaMemcpyHostToDevice);
	cudaMalloc(&ctx.dP, ctx.width * sizeof(float));
	cudaMalloc(&ctx.dP2, ctx.width * sizeof(float));
	cudaMemcpyToSymbol(DM, ctx.M, ctx.maskWidth * sizeof(float));


	begin = std::chrono::high_resolution_clock::now();
	convolution1DGPU<<<ceil(ctx.width / 256.0), TILE_SIZE>>>(ctx.dN, ctx.dP,
		ctx.maskWidth, ctx.width);
	end = std::chrono::high_resolution_clock::now();
	auto elapsedGPU = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
	std::cout << elapsedGPU.count() << " microseconds" << std::endl;

	begin = std::chrono::high_resolution_clock::now();
	tiledConvolution1D<<<ceil(ctx.width / 256.0), TILE_SIZE>>>(ctx.dN, ctx.dP2,
		ctx.maskWidth, ctx.width);
	end = std::chrono::high_resolution_clock::now();
	auto elapsedGPU2 = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
	std::cout << elapsedGPU2.count() << " microseconds" << std::endl;

	cudaDeviceSynchronize();

	cudaMemcpy(ctx.P2, ctx.dP, ctx.width * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ctx.P3, ctx.dP2, ctx.width * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(ctx.dN);
	cudaFree(ctx.dM);
	cudaFree(ctx.dP);
	cudaFree(ctx.dP2);

	compareCPUToGPU(&ctx);

	// CPU to GPU (1st Version)
	auto speedUp = std::chrono::duration_cast<std::chrono::microseconds>(elapsedCPU).count() / elapsedGPU.count();
	std::cout << "CPU to GPU (1st Version) speedUp " << speedUp  << "X Speedup" << std::endl;

	speedUp = std::chrono::duration_cast<std::chrono::microseconds>(elapsedCPU).count() / elapsedGPU2.count();
	std::cout << "CPU to GPU (2nd Version) speedUp " << speedUp << "X Speedup" << std::endl;

	float speedUpFloat = (float) elapsedGPU.count() / (float) elapsedGPU2.count();
	printf("%f\n", speedUpFloat);
	std::cout << "GPU (1st Version) to GPU (2nd Version) speedUp " << speedUpFloat << "X Speedup" << std::endl;

	// foo(&ctx);
	return 0;
}