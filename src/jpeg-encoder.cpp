#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "util.h"


#define COEFFS(Cu,Cv,u,v) { \
	if (u == 0) Cu = 1.0f / sqrtf(2.0f); else Cu = 1.0f; \
	if (v == 0) Cv = 1.0f / sqrtf(2.0f); else Cv = 1.0f; \
	}

#define PI 3.141592653589793f

/* JPEG (sequential) encoding
	Color space transformation : Checked!
	Block splitting (no padding for now) : 
	Discrete cosine transform
	Quantization
	Entropy coding
*/
 
// TODO: 	Improve Performance
// 			Create a CUDA version.

// O(n^4)
void dct(float inputBlock[8][8], float outputBlock[8][8]) {

	int u, v, x, y;

	for (u = 0; u < 8; u++) {

		for (v = 0; v < 8; v++) {

			float cU, cV, sum = 0.0f;
			COEFFS(cU, cV, u, v);

			for (x = 0; x < 8; x++) {

				for (y = 0; y < 8; y++) {

					sum += inputBlock[x][y] * 
					cosf((2.0f * x + 1.0f) * u * PI / 16.0f) *
					cosf((2.0f * y + 1.0f) * v * PI / 16.0f);
				}	
			}

			outputBlock[u][v] = 0.25f * cU * cV * sum;
		}
	}
}

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

void quantizeBlock(	float dctBlock[8][8], 
					int quantizedBlock[8][8],
					const int qTable[8][8]) 
{

	for (int u = 0; u < 8; u++) {
		for (int v = 0; v < 8; v++) {
			float quantized = dctBlock[u][v] / qTable[u][v];
			quantizedBlock[u][v] = (int)roundf(quantized);
		}
	}

}

const int ZIGZAG_INDEX[64][2] = {
    {0,0}, {0,1}, {1,0}, {2,0}, {1,1}, {0,2}, {0,3}, {1,2},
    {2,1}, {3,0}, {4,0}, {3,1}, {2,2}, {1,3}, {0,4}, {0,5},
    {1,4}, {2,3}, {3,2}, {4,1}, {5,0}, {6,0}, {5,1}, {4,2},
    {3,3}, {2,4}, {1,5}, {0,6}, {0,7}, {1,6}, {2,5}, {3,4},
    {4,3}, {5,2}, {6,1}, {7,0}, {7,1}, {6,2}, {5,3}, {4,4},
    {3,5}, {2,6}, {1,7}, {2,7}, {3,6}, {4,5}, {5,4}, {6,3},
    {7,2}, {7,3}, {6,4}, {5,5}, {4,6}, {3,7}, {4,7}, {5,6},
    {6,5}, {7,4}, {7,5}, {6,6}, {5,7}, {6,7}, {7,6}, {7,7}
};

// Flattens your 2D quantized block array into a 1D sequence
void zigzagScanBlock(int qInputBlock[8][8], int outputVector[64]) {
    for (int i = 0; i < 64; i++) {
        int r = ZIGZAG_INDEX[i][0];
        int c = ZIGZAG_INDEX[i][1];
        outputVector[i] = qInputBlock[r][c];
    }
}

int main() {

	char * baseImgDir = "kodak-image-suite/PhotoCD_PCD0992/";
	char * imgName = "19.png";
	char imgPath[100] = {0};
	strcat(imgPath, baseImgDir);
	strcat(imgPath, imgName);

	int x, y, n;
	unsigned char * imgData = stbi_load(imgPath, &x, &y, &n, 0);
	if (imgData == NULL) {
		printf("Error loading an image %s\n", imgPath);
		exit(-1);
	}

	printf("%d scanlines of %d pixels\n", y, x);
	printf("Each pixel consisting of %d interleaved 8-bit components \n", n);
	printf("%ld\n", sizeof(imgData));


	// Color Space Transformation.
	float yCrCbImgData[y * x * 3] = {0};
	for (int i = 0; i < y * x * 3; i += 3) {
		
		float r = imgData[i];
		float g = imgData[i + 1];
		float b = imgData[i + 2];

		float y  =  0.29900f * r + 0.58700f * g + 0.11400f * b;
		// 128.0f?
		float cB = -0.16874f * r - 0.33126f * g + 0.50000f * b + 128.0f;
		float cR =  0.50000f * r - 0.41869f * g - 0.08131f * b + 128.0f;

		yCrCbImgData[i] = y;
		yCrCbImgData[i + 1] = cB;
		yCrCbImgData[i + 2] = cR;

		if (i < 15) {
			printf("Pixel %d -> Y: %f, Cb: %f, Cr: %f\n", i/3, y, cB, cR);
		}
	}


	float blockY[8][8];
	float blockCb[8][8];
	float blockCr[8][8];
	
	float dctBlockY[8][8];
	float dctBlockCb[8][8];
	float dctBlockCr[8][8];

	int quantizedBlockY[8][8];
	int quantizedBlockCb[8][8];
	int quantizedBlockCr[8][8];

	int zigzagVectorY[64]  = {0};
	int zigzagVectorCb[64] = {0};
	int zigzagVectorCr[64] = {0};

	char * outputPath = "output.jpg";
	JpegEncoder encoder;
	jpegEncoderInit(&encoder, outputPath, x, y, Q_LUMINANCE, Q_CHROMINANCE);

	for (int blockYStart = 0; blockYStart < y; blockYStart += 8) {
		for (int blockXStart = 0; blockXStart < x; blockXStart += 8) {

			for (int row = 0; row < 8; row++) {
				for (int col = 0; col < 8; col++) {


					int currentPixelX = blockXStart + col;
					int currentPixelY = blockYStart + row;
					int pixelIdx = (currentPixelY * x + currentPixelX) * 3;

	                blockY[row][col]  = yCrCbImgData[pixelIdx]     - 128.0f;
                	blockCb[row][col] = yCrCbImgData[pixelIdx + 1] - 128.0f;
                	blockCr[row][col] = yCrCbImgData[pixelIdx + 2] - 128.0f;
				}
			}



			// DFT
			dct(blockY, dctBlockY);
			dct(blockCb, dctBlockCb);
			dct(blockCr, dctBlockCr);


			// Quntization
			quantizeBlock(dctBlockY, quantizedBlockY, Q_LUMINANCE);
			quantizeBlock(dctBlockCb, quantizedBlockCb, Q_CHROMINANCE);
			quantizeBlock(dctBlockCr, quantizedBlockCr, Q_CHROMINANCE);


			// Zigzag
			zigzagScanBlock(quantizedBlockY,  zigzagVectorY);
			zigzagScanBlock(quantizedBlockCb, zigzagVectorCb);
			zigzagScanBlock(quantizedBlockCr, zigzagVectorCr);

			// Entropy coding (Huffman) straight into the file
			jpegEncodeBlock(&encoder, zigzagVectorY,  0);
			jpegEncodeBlock(&encoder, zigzagVectorCb, 1);
			jpegEncodeBlock(&encoder, zigzagVectorCr, 2);

			// Huffman
			


			// Writing to file. (AI)

		}

		if (blockYStart == 0) {
			// for (int i = 0; i < 8; i++) {
			// 	for (int j = 0; j < 8; j++) {
			// 		printf("%f ", quantizedBlockY[i][j]);
			// 	}
			// printf("\n");
			// }

			// printf("Zig-zag vector values:\n");
			// for(int i = 0; i < 64; i++) {
    		// 	printf("%d ", zigzagVectorY[i]);
			// }
			// printf("\n");
		}
	}

	jpegEncoderFinish(&encoder);
	printf("Wrote %s\n", outputPath);

	return 0;
}