#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



/* JPEG (sequential) encoding
	Color space transformation : Checked!
	Block splitting (no padding for now) : Checked
	Discrete cosine transform : Checked
	Quantization : Checked
	Entropy coding (Optional)
*/
 
// TODO: 	Improve Performance
// 			Create a CUDA version.

// O(n^4)
void dct(float block[8][8]) {

	const float PI = 3.141592653589793f;
	int u, v, x, y;
	float temp[8][8];

	for (v = 0; v < 8; v++) {
		for (u = 0; u < 8; u++) {

			float sum = 0.0f;
			float cu = (u == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;
			float cv = (v == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;

			for (y = 0; y < 8; y++) {
				for (x = 0; x < 8; x++) {
					sum += block[y][x] *
					cosf((2.0f * x + 1.0f) * u * PI / 16.0f) *
					cosf((2.0f * y + 1.0f) * v * PI / 16.0f);
				}
			}

			temp[v][u] = 0.25f * cu * cv * sum;
		}
	}

	for (v = 0; v < 8; v++) {
		for (u = 0; u < 8; u++) {
			block[v][u] = temp[v][u];
		}
	}
}

void idct(float block[8][8]) {

	const float PI = 3.141592653589793f;
	int u, v, x, y;
	float temp[8][8];

	for (v = 0; v < 8; v++) {
		for (u = 0; u < 8; u++) {

			float sum = 0.0f;
			float cu = (u == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;
			float cv = (v == 0) ? 1.0f / sqrtf(2.0f) : 1.0f;

			for (y = 0; y < 8; y++) {
				for (x = 0; x < 8; x++) {
					sum += block[y][x] * cu * cv *
					cosf((2.0f * x + 1.0f) * u * PI / 16.0f) *
					cosf((2.0f * y + 1.0f) * v * PI / 16.0f);
				}
			}

			temp[v][u] = 0.25f * sum;
		}
	}

	for (v = 0; v < 8; v++) {
		for (u = 0; u < 8; u++) {
			block[v][u] = temp[v][u];
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

void quantizeBlock(float block[8][8], int channel) 
{
	float temp[8][8];
	int u, v;
	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			float quantizedFactor = (channel == 0) ? 
			Q_LUMINANCE[u][v] : Q_CHROMINANCE[u][v];
			float quantized = block[u][v] / quantizedFactor;
			temp[u][v] = (int)roundf(quantized);
		}
	}

	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			block[u][v] = temp[u][v];
		}
	}

}

void dequantizeBlock(float block[8][8], int channel) {
		float temp[8][8];
	int u, v;
	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			float quantizedFactor = (channel == 0) ? 
			Q_LUMINANCE[u][v] : Q_CHROMINANCE[u][v];
			float quantized = block[u][v] * quantizedFactor;
			temp[u][v] = (int)roundf(quantized);
		}
	}

	for (u = 0; u < 8; u++) {
		for (v = 0; v < 8; v++) {
			block[u][v] = temp[u][v];
		}
	}
}

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

		// Debugging.
		// if (i < 15) {
		// 	printf("Pixel %d -> Y: %f, Cb: %f, Cr: %f\n", i/3, Y, Cb, Cr);
		// }
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

void encodeJPEG(const char *imgPath, const char *imgName) {


	int width, height, nCh;
	unsigned char * imgData = stbi_load(imgPath, &width, &height, &nCh, 0);
	if (imgData == NULL) {
		printf("Error loading an image %s\n", imgPath);
		exit(-1);
	}

	// printf("%d scanlines of %d pixels\n", height, width);
	// printf("Each pixel consisting of %d interleaved 8-bit components \n", nCh);

	// Color Space Transformation.
	unsigned char *YCbCr = (unsigned char *) malloc((width * height * nCh) * sizeof(unsigned char));
	RGBToYCbCr(imgData, YCbCr, (width * height * nCh));

	// char * outputPath = "output.jpg";
	// JpegEncoder encoder;
	// jpegEncoderInit(&encoder, outputPath, x, y, Q_LUMINANCE, Q_CHROMINANCE);

	unsigned char *compressedData = (unsigned char *) malloc((width * height * nCh) * sizeof(unsigned char));
	int channel, blockYStart, blockXStart;
	for (channel = 0; channel < 3; channel++) {
		for (blockYStart = 0; blockYStart < height; blockYStart += 8) {
			for (blockXStart = 0; blockXStart < width; blockXStart += 8) {

				float block[8][8];

				// Block Splitting
				int x, y;
				for (y = 0; y < 8; y++) {
					for (x = 0; x < 8; x++) {
						int idx = ((blockYStart + y) * width + (blockXStart + x)) * 3 + channel;
						block[y][x] = (float)(YCbCr[idx] - 128.0f); 
					}
				}

				// Discrete Cosine Transform.
				dct(block);

				// Debugging.
				// if (blockXStart == 0 && blockYStart == 0) {
				// 	for (int i = 0; i < 5; i++) {
				// 		printf("%f ", block[0][i]);
				// 	}
				// 	printf("\n");
				// }

				// Basic Quantization
				quantizeBlock(block, channel);

				// Decode.
				dequantizeBlock(block, channel);
				

				// Debugging.
				// if (blockXStart == 0 && blockYStart == 0) {
				// 	for (int i = 0; i < 5; i++) {
				// 		printf("%f ", block[0][i]);
				// 	}
				// 	printf("\n");
				// }



				// Inverse DCT
				idct(block);

				// Store the compressed data
				for (y = 0; y < 8; y++) {
					for (x = 0; x < 8; x++) {
						int idx = ((blockYStart + y) * width + (blockXStart + x)) * 3 + channel;
						float value = block[y][x] + 128.0f;
						compressedData[idx] = (unsigned char)std::min(255.0f, std::max(0.0f, value));
					}
				}
			}	

		}
	}

	// Huffman coding.


	// Convert back to RGB
	unsigned char *outputRGB = (unsigned char*) malloc((width * height * nCh) * sizeof(unsigned char));
	YCbCrToRGB(compressedData, outputRGB, (width * height * nCh));

	char outputPath[200] = {0};
	strcat(outputPath, imgName);
	strcat(outputPath, "-encoded.jpg");
	stbi_write_jpg(outputPath, width, height, 3, outputRGB, 90);

	stbi_image_free(imgData);
}

int main() {

	const char *baseImgDir = "kodak-image-suite/PhotoCD_PCD0992/";

	struct dirent *entry;
	DIR *dir = opendir(baseImgDir);

	if (dir == NULL) {
		perror("opendir: ");
		exit(-1);
	}

	while ((entry = readdir(dir)) != NULL) {

		if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
		printf("%s\n", entry->d_name);

		char imgPath[100] = {0};
		strcat(imgPath, baseImgDir);
		strcat(imgPath, entry->d_name);

		encodeJPEG(imgPath, entry->d_name);
	}

	return 0;
}