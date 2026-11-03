#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <algorithm>
#include <chrono>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Slower.

/* JPEG (sequential) encoding
	Color space transformation : Checked!
	Block splitting (no padding for now) : Checked
	Discrete cosine transform : Checked
	Quantization : Checked
	Entropy coding (Optional)
*/
 
// TODO: 	Improve Performance
// 			Create a CUDA version.


#define WTIH_DECODING

#define WITH_CUDA

#define JPEG_BLOCK_WIDTH 8
#define JPEG_BLOCK_HEIGHT 8


// Standard JPEG Luminance (Y) Quantization Table
const int Q_LUMINANCE[8][8] = {
    {16, 11, 10, 16,  24,  40,  51,  61},
    {12, 12, 14, 19,  26,  58,  60,  55},
    {14, 13, 16, 24,  40,  57,  69,  56},
    {14, 17, 22, 29,  51,  87,  80,  62},
    {18, 22, 37, 56,  68, 109, 103,  77},
    {24, 35, 55, 64,  81, 104, 113,  92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103,  99}
};

// Standard JPEG Chrominance (Cb/Cr) Quantization Table
const int Q_CHROMINANCE[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};


// Precompute cosine bases
static const float COS_TABLE[8][8] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, // u = 0
    {0.980785251f, 0.831469595f, 0.555570185f, 0.195090234f, -0.195090324f, -0.555570364f, -0.831469655f, -0.98078531f}, // u = 1
    {0.923879504f, 0.382683426f, -0.382683516f, -0.923879623f, -0.923879504f, -0.382683128f, 0.382683605f, 0.923879564f}, // u = 2
    {0.831469595f, -0.195090324f, -0.98078531f, -0.555570006f, 0.555570424f, 0.980785251f, 0.19509007f, -0.831469774f}, // u = 3
    {0.707106769f, -0.707106769f, -0.70710665f, 0.707107008f, 0.707106769f, -0.707107246f, -0.707106531f, 0.707106829f}, // u = 4
    {0.555570185f, -0.98078531f, 0.195090413f, 0.831469357f, -0.831469774f, -0.195090219f, 0.98078531f, -0.555570841f}, // u = 5
    {0.382683426f, -0.923879504f, 0.923879564f, -0.382683903f, -0.382682979f, 0.923879325f, -0.923879743f, 0.382683903f}, // u = 6
    {0.195090234f, -0.555570006f, 0.831469357f, -0.980785191f, 0.980785429f, -0.831469595f, 0.555571139f, -0.195091546f} // u = 7
};

// Decoding
// --------------------------

void idctNaive(float in[64], float out[64]) {

	const float PI = 3.141592653589793f;
	int u, v, x, y;
	for (y = 0; y < 8; y++)
	for (x = 0; x < 8; x++)
	{

		float sum = 0.0f;

		for (v = 0; v < 8; v++)
		for (u = 0; u < 8; u++)
		{
			float Cv = (v == 0) ? 1 / sqrtf(2.0f) : 1.0f;
			float Cu = (u == 0) ? 1 / sqrtf(2.0f) : 1.0f;

			sum += in[v * 8 + u] * Cv * Cu *
			cosf((2.0f * y + 1.0f) * v * PI / 16.0f) *
			cosf((2.0f * x + 1.0f) * u * PI / 16.0f);
		}

		out[y * 8 + x] = 0.25f * sum;
	}
}


void deQuantizeBlock(float in[64], float out[64],int channel) {

	int u, v;
	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			float quantizedFactor = (channel == 0) ? 
			Q_LUMINANCE[u][v] : Q_CHROMINANCE[u][v];
			float quantized = in[u * 8 + v] * quantizedFactor;
			out[u * 8 + v] = (int)roundf(quantized);
		}
	}
}

// --------------------------

// Encoding
// --------------------------

// Naive Version
void dctNaive(float in[64], float out[64]) {

	const float PI = 3.141592653589793f;
	int u, v, x, y;
	for (v = 0; v < 8; v++)
	for (u = 0; u < 8; u++)
	{

		float sum = 0.0f;
		float Cv = (v == 0) ? 1 / sqrtf(2.0f) : 1.0f;
		float Cu = (u == 0) ? 1 / sqrtf(2.0f) : 1.0f;

		for (y = 0; y < 8; y++)
		for (x = 0; x <8; x++)
		{
			sum += in[y * 8 + x] * 
			cosf((2.0f * x + 1.0f) * u * PI / 16.0f) *
			cosf((2.0f * y + 1.0f) * v * PI / 16.0f);
		}

		out[v * 8 + u] = 0.25f * Cv * Cu * sum;
	}
}




void dct1DPrecompute(float in[8], float out[8]) {
	
	int u, x;
	for (u = 0; u < 8; u++) {

		float sum = 0.0f;
		float Cu = (u == 0) ? 1 / sqrtf(2.0f) : 1.0f;

		for (x = 0; x < 8; x++) {
			sum += in[x] * COS_TABLE[u][x];
		}

		out[u] = 0.5f * Cu * sum;
	}
}

void dct2DAs1D(float in[64], float out[64]) {

	int i, j;
	float in1D[8], out1D[8], temp[64];

	// transform row
	for (j = 0; j < 8; j++) {
		for (i = 0; i < 8; i++) in1D[i] = in[j * 8 + i];
		dct1DPrecompute(in1D, out1D);
		for (i = 0; i < 8; i++) temp[j * 8 + i] = out1D[i];
	}

	for (j = 0; j < 8; j++) {
		for (i = 0; i < 8; i++) in1D[i] = temp[i * 8 + j];
		dct1DPrecompute(in1D, out1D);
		for (i = 0; i < 8; i++) out[i * 8 + j] = out1D[i];
	}
}


void quantizeBlock(float in[64], float out[64],int channel) {

	int u, v;
	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			float quantizedFactor = (channel == 0) ?
			Q_LUMINANCE[u][v] : Q_CHROMINANCE[u][v];
			float quantized = in[u * 8 + v] / quantizedFactor;
			out[u * 8 + v] = (int)roundf(quantized);
		}
	}
}

// --------------------------

// Compression Statistics
// --------------------------

// Zigzag scan order (low -> high frequency), same as real JPEG.
static const int ZIGZAG[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// Count zero coefficients and estimate the entropy-coded size of one
// quantized block: zigzag scan + run-length of zeros, then ~1 byte per
// (run, value) symbol plus the bytes needed for the value itself.
// This is a rough estimate -- real Huffman coding would do a bit better.
void analyzeQuantizedBlock(const float block[64], long &zeroCount, long &rleBytes) {

	int i;
	for (i = 0; i < 64; i++) {
		if ((int)block[i] == 0) zeroCount++;
	}

	int run = 0;
	rleBytes += 2; // DC coefficient: ~1 byte symbol + ~1 byte amplitude (rough)
	for (i = 1; i < 64; i++) {

		int coef = (int)block[ZIGZAG[i]];
		if (coef == 0) {
			run++;
			continue;
		}

		rleBytes += run / 16; // ZRL marker for every 16 zeros in a run
		run = 0;

		int magnitude = abs(coef), sizeBits = 0;
		while (magnitude) { sizeBits++; magnitude >>= 1; }
		rleBytes += 1 + (sizeBits + 7) / 8; // RLE symbol + amplitude bytes
	}
	if (run > 0) rleBytes += 1; // EOB marker
}

// --------------------------


// Color Space Transformation
// --------------------------------------
void RGBToYCbCr(unsigned char *RGB, unsigned char *YCbCr, int n) {

	int i;
	for (i = 0; i < n; i += 3) {

		unsigned char r = RGB[i];
		unsigned char g = RGB[i + 1];
		unsigned char b = RGB[i + 2];


		unsigned char Y  =  (unsigned char)(0.29900f * r + 0.58700f * g + 0.11400f * b);
		unsigned char Cb = (unsigned char)(-0.16874f * r - 0.33126f * g + 0.50000f * b + 128.0f);
		unsigned char Cr =  (unsigned char) (0.50000f * r - 0.41869f * g - 0.08131f * b + 128.0f);

		YCbCr[i] = Y;
		YCbCr[i + 1] = Cb;
		YCbCr[i + 2] = Cr;
	}
}	

void YCbCrToRGB(unsigned char *YCbCr, unsigned char *RGB, int n) {

	int i;
	for (i = 0; i < n; i += 3) {

		unsigned char Y = YCbCr[i];
		unsigned char Cb = YCbCr[i + 1];
		unsigned char Cr = YCbCr[i + 2];

		float R = Y + 1.402f * (Cr - 128.0f);
		float G = Y - 0.344136f * (Cb - 128.0f) - 0.714136f * (Cr - 128.0f);
		float B = Y + 1.772f * (Cb - 128.0f);

		// // Debugging.
		// if (i < 15) {
		// 	printf("Pixel %d -> Y: %f, Cb: %f, Cr: %f\n", i/3, Y, Cb, Cr);
		// }

		RGB[i] = (unsigned char)std::min(255.0f, std::max(0.0f, R));
		RGB[i + 1] = (unsigned char)std::min(255.0f, std::max(0.0f, G));
		RGB[i + 2] = (unsigned char)std::min(255.0f, std::max(0.0f, B));
	}
}

// --------------------------------------


void encodeJPEG(const char *imgPath, const char *imgName) {


	int width, height, nCh;
	unsigned char * imgData = stbi_load(imgPath, &width, &height, &nCh, 0);
	if (imgData == NULL) {
		printf("Error loading an image %s\n", imgPath);
		exit(-1);
	}

	// Color Space Transformation.
	unsigned char *YCbCr = (unsigned char *) malloc((width * height * nCh) * sizeof(unsigned char));
	RGBToYCbCr(imgData, YCbCr, (width * height * nCh));

	unsigned char *compressedData = (unsigned char *) malloc((width * height * nCh) * sizeof(unsigned char));
	int channel, blockYStart, blockXStart;

	long totalCoeffs = 0, zeroCoeffs = 0, rleBytes = 0;

	for (channel = 0; channel < 3; channel++) {
		for (blockYStart = 0; blockYStart < height; blockYStart += 8) {
			for (blockXStart = 0; blockXStart < width; blockXStart += 8) {

				float block[64];
				
				// Block Splitting
				int x, y;
				for (y = 0; y < 8; y++) {
					for (x = 0; x < 8; x++) {
						int idx = ((blockYStart + y) * width + (blockXStart + x)) * 3 + channel;
						block[y * 8 + x] 	= (float)(YCbCr[idx] - 128.0f);
					}
				}

				// Encoding 
				// ---------------------------------------
				// Discrete Cosine Transform.
				float dctBlock[64];
				dct2DAs1D(block, dctBlock);

				// Quantization
				float quantizedBlock[64];
				quantizeBlock(dctBlock, quantizedBlock, channel);

				// Compression statistics
				totalCoeffs += 64;
				analyzeQuantizedBlock(quantizedBlock, zeroCoeffs, rleBytes);
				// ---------------------------------------


				#ifdef WTIH_DECODING
				// Decoding (Simple)
				// ---------------------------------------
				float deQuantizedBlock[64];
				deQuantizeBlock(quantizedBlock, deQuantizedBlock, channel);
				idctNaive(deQuantizedBlock, block);
				// ---------------------------------------
				#endif

				// Store the compressed data
				for (y = 0; y < 8; y++) {
					for (x = 0; x < 8; x++) {
						int idx = ((blockYStart + y) * width + (blockXStart + x)) * 3 + channel;
						#ifdef WTIH_DECODING
						float value = block[y * 8 + x] + 128.0f;
						#else
						float value = quantizedBlock[y * 8 + x] + 128.0f;
						#endif
						compressedData[idx] = (unsigned char)std::min(255.0f, std::max(0.0f, value));
					}
				}

			}	

		}
	}

	unsigned char *outputRGB = (unsigned char*) malloc((width * height * nCh) * sizeof(unsigned char));
	YCbCrToRGB(compressedData, outputRGB, (width * height * nCh));

	char outputPath[200] = {0};
	strcat(outputPath, "output/");
	strcat(outputPath, imgName);
	strcat(outputPath, "-compressed.png");
	stbi_write_png(outputPath, width, height, 3, outputRGB, width * 3);

	// Compression statistics summary
	struct stat st;
	long origBytes = (stat(imgPath, &st) == 0) ? (long)st.st_size : 0;
	printf("  zero coefficients: %ld / %ld (%.1f%%)\n",
		zeroCoeffs, totalCoeffs, 100.0 * zeroCoeffs / totalCoeffs);
	printf("  est. entropy-coded size: %ld bytes", rleBytes);
	if (origBytes > 0) {
		printf("  vs original %ld bytes  ->  %.1fx smaller\n",
			origBytes, (double)origBytes / rleBytes);
	} else {
		printf("\n");
	}

	stbi_image_free(imgData);
	free(outputRGB);
	free(compressedData);
}

int main() {

	const char *baseImgDir = "selected-pictures/";

	struct dirent *entry;
	DIR *dir = opendir(baseImgDir);

	if (dir == NULL) {
		perror("opendir: ");
		exit(-1);
	}

	const int BLOCK_SIZE = 8;

	int i = 0;
	while ((entry = readdir(dir)) != NULL) {

		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
		
		if (strcmp(entry->d_name, "readme.txt") == 0) {
			continue;
		}
		// printf("%s\n", entry->d_name);

		char imgPath[100] = {0};
		strcat(imgPath, baseImgDir);
		strcat(imgPath, entry->d_name);
		
		#ifdef WITH_CUDA

		#else
		// CPU
		auto begin = std::chrono::high_resolution_clock::now();
		encodeJPEG(imgPath, entry->d_name);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsedCPU = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		
		if (i < 250000) {
			std::cout << imgPath << std::endl;
			std::cout << elapsedCPU.count() / 1000.0f  << " seconds" << std::endl;
			i++;
		} else {
			 break;
		}

		#endif
	}

	return 0;
}