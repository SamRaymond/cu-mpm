#ifndef MPM_COMMON_H
#define MPM_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>

//------------------------------------------------------------
// Grid / domain parameters
//------------------------------------------------------------
static const int   NX = 512;       // number of cells in x-direction
static const int   NY = 64;        // number of cells in y-direction
static const float DX = 0.1f;      // cell size in x
static const float DY = 0.1f;      // cell size in y

//------------------------------------------------------------
// Particle / simulation parameters
//------------------------------------------------------------
static const int   NUM_PARTICLES = 100000;  // total particles
static const float DT            = 1e-4f; // timestep
static const int   MAX_ITER      = 200000;
static const int   SAVE_INTERVAL = 250;  // write .vtp every SAVE_INTERVAL steps

static const int   BLOCK_SIZE    = 1024;

//------------------------------------------------------------
// CUDA Error Macro
//------------------------------------------------------------
#define CUDA_CHECK( err )  { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#endif // MPM_COMMON_H
