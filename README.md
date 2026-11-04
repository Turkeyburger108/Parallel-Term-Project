# Parallel Term Project


## Folder Structure


+ Parallel Term Project
  + src
  	+ jpeg-codec.cu **the main program**
  	+ stb_image.h **helper library for retrieving RGB components**
  	+ stb_image_write.h **helper library for writing to an image file**
  + selected-pictures **actual images used for program**
  + cuda-samples
  + image-datasets

## How to run.

### Prerequisite

CUDA - Installation guide for 
[Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and
[Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### Program's Flag

Flags in jpeg-codec.cu have the form `#define ....`
you can comment or uncomment to enable certain features.

`WITH_ANALYSIS // Display the amount of entropy compressed` \
`WTIH_DECODING // Include Dequantization and Inverse DCT` \
`WITH_CUDA // Switch to CUDA implementation` \
`WITH_TIME_PER_IMAGE // Display the time it takes to operate 1 image` \


### How to compile

1. Go to the root folder
2. Get your GPU code by running `nvcc --list-gpu-code`
3. run ` nvcc -arch=<Your GPU code> -o <Your Program Name> src/jpeg-codec.cu `
4. run the executable from the root folder
