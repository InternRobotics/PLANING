#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
 

// #define MAX_NB_POINTS 8  // org
#define MAX_NB_POINTS 3                   
 

#define MAX_TILE_TOUCHED 10800

#endif