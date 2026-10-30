#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cmath>
#include <random>
#include <chrono>
#include <iostream>

// Convolution


typedef struct {
	int width;
	int maskWidth;

	float *N;
	float *M;
	float *P;

	float *dN;
	float *dM;
	float *dP;
	float *P2;

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

__global__ void convolution1DGPU(float *N, float *M, float *P,
	int maskWidth, int width) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float pValue = 0;
	int nStartPoint = idx - (maskWidth / 2);
	for (int j = 0; j < maskWidth; j++) {
		if (nStartPoint + j >= 0 && nStartPoint + j < width) {
			pValue += N[nStartPoint + j] * M[j];
		}
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

int main() {

	Context ctx;
	ctx.width = 1 << 20;
	ctx.maskWidth = 7;

	ctx.N 	= 	(float *) malloc(ctx.width 		* sizeof(float));
	ctx.M 	= 	(float *) malloc(ctx.maskWidth 	* sizeof(float));
	ctx.P 	= 	(float *) malloc(ctx.width 		* sizeof(float));
	ctx.P2 	= 	(float *) malloc(ctx.width 		* sizeof(float));


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


	begin = std::chrono::high_resolution_clock::now();
	convolution1DGPU<<<ceil(ctx.width / 256.0), 256>>>(ctx.dN, ctx.dM, ctx.dP,
		ctx.maskWidth, ctx.width);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	auto elapsedGPU = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
	std::cout << elapsedGPU.count() << " microseconds" << std::endl;


	cudaMemcpy(ctx.P2, ctx.dP, ctx.width * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(ctx.dN);
	cudaFree(ctx.dM);
	cudaFree(ctx.dP);

	auto speedUp = std::chrono::duration_cast<std::chrono::microseconds>(elapsedCPU).count() / elapsedGPU.count();
	std::cout << speedUp << "X Speedup" << std::endl;

	// foo(&ctx);
	return 0;
}