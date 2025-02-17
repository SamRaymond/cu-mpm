#ifndef MPM_KERNELS_CUH
#define MPM_KERNELS_CUH

#include "mpm_common.h"
#include "mpm_data.h"

// 1D Basic GIMP shape function (device)
__device__ float shapeGIMP1D(float xParticle, float xNode, float halfSize, float dx);

// 2D GIMP shape function = product of shape in x and y
__device__ float shapeGIMP2D(float xp, float yp,
                             float halfSizeX, float halfSizeY,
                             int nx, int ny, float dx, float dy);

// Kernels
__global__
void InitParticlesKernel(MpmParticleData pData, int numParticles, float vxInit, float vyInit);

__global__
void ComputeCellIDKernel(const MpmParticleData pData, int* outCellID, int numParticles);

__global__
void buildCellRanges(const int* __restrict__ cellIDSorted,
                     int* __restrict__ cellStart,
                     int* __restrict__ cellEnd,
                     int numParticles);

__global__
void P2GKernelScatter(MpmParticleData pData,
                      MpmGridNodeData gData,
                      const int* __restrict__ cellStart,
                      const int* __restrict__ cellEnd,
                      const int* __restrict__ particleIndices,
                      int numCells);

__global__
void GridUpdateKernel(MpmGridNodeData gData, int totalNodes);

__global__
void G2PKernel(MpmParticleData pData,
               MpmGridNodeData gData,
               const int* __restrict__ cellID,
               int numParticles);

__global__
void ParticleUpdateKernel(MpmParticleData pData, int numParticles, float dt);

#endif // MPM_KERNELS_CUH
