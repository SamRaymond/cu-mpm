#include "mpm_kernels.cuh"

// ----------------------------------------------------------
// 1D Basic GIMP shape function
// ----------------------------------------------------------
__device__
float shapeGIMP1D(float xParticle, float xNode, float halfSize, float dx)
{
    float r     = (xParticle - xNode) / dx;   // center relative to node
    float alpha = halfSize / dx;              // dimensionless half-size

    float left  = -1.0f - alpha;
    float c1    = -1.0f + alpha;
    float c2    =  1.0f - alpha;
    float right =  1.0f + alpha;

    if (r <= left) {
        return 0.0f;
    } else if (r <= c1) {
        float tmp = (r + 1.0f + alpha);
        return (tmp * tmp) / (4.0f * alpha);
    } else if (r <= 0.0f) {
        return 1.0f + (r / alpha) + (0.5f * (r * r) / (alpha * alpha));
    } else if (r < c2) {
        return 1.0f - (r / alpha) + (0.5f * (r * r) / (alpha * alpha));
    } else if (r < right) {
        float tmp = (1.0f + alpha - r);
        return (tmp * tmp) / (4.0f * alpha);
    } else {
        return 0.0f;
    }
}

// ----------------------------------------------------------
// 2D GIMP shape = product of shape in x and shape in y
// ----------------------------------------------------------
__device__
float shapeGIMP2D(float xp, float yp,
                  float halfSizeX, float halfSizeY,
                  int nx, int ny,
                  float dx, float dy)
{
    float xNode = nx * dx;
    float yNode = ny * dy;

    float sx = shapeGIMP1D(xp, xNode, halfSizeX, dx);
    float sy = shapeGIMP1D(yp, yNode, halfSizeY, dy);
    return sx * sy;
}

// ----------------------------------------------------------
// Kernel: Initialize Particles
// ----------------------------------------------------------
__global__
void InitParticlesKernel(MpmParticleData pData, int numParticles, float vxInit, float vyInit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    // Domain dimensions
    float domainX = NX * DX;
    float domainY = NY * DY;

    // Set the height of the particle block to 20% of the domain height
    float blockHeight = 0.2f * domainY;
    float blockWidth = blockHeight;  // Square block

    // Compute particle spacing (fixed at half a cell)
    float spacingX = DX / 2.0f;
    float spacingY = DY / 2.0f;

    // Compute the number of particles in each direction
    int numX = (int)(blockWidth / spacingX);
    int numY = (int)(blockHeight / spacingY);

    // Ensure we do not exceed the total number of particles
    if (numX * numY > numParticles) {
        numY = numParticles / numX;  // Adjust to fit
    }

    // Compute the starting position (centered in domain)
    float startX = 0.5f * (domainX - (numX * spacingX));
    float startY = 0.5f * (domainY - (numY * spacingY));

    // Compute grid indices
    int px = tid % numX;  // Column index
    int py = tid / numX;  // Row index

    // Ensure we do not place extra particles
    if (px >= numX || py >= numY) return;

    // Assign structured grid positions
    pData.pos_x[tid] = startX + px * spacingX;
    pData.pos_y[tid] = startY + py * spacingY;

    // Initialize velocities, mass, and momentum
    pData.vel_x[tid] = vxInit;
    pData.vel_y[tid] = vyInit;
    pData.mass[tid]  = 1.0f;
    pData.mom_x[tid] = pData.mass[tid] * vxInit;
    pData.mom_y[tid] = pData.mass[tid] * vyInit;

    // Set particle half-sizes (for GIMP)
    pData.halfSizeX[tid] = 0.5f * DX;
    pData.halfSizeY[tid] = 0.5f * DY;

    // Initialize stress, strain, and velocity gradients
    pData.stress_xx[tid] = 0.f;
    pData.stress_yy[tid] = 0.f;
    pData.stress_xy[tid] = 0.f;
    pData.strain_xx[tid] = 0.f;
    pData.strain_yy[tid] = 0.f;
    pData.strain_xy[tid] = 0.f;
    pData.grad_vx_x[tid] = 0.f;
    pData.grad_vx_y[tid] = 0.f;
    pData.grad_vy_x[tid] = 0.f;
    pData.grad_vy_y[tid] = 0.f;
}

// ----------------------------------------------------------
// Kernel: Compute Cell ID for each particle
// ----------------------------------------------------------
__global__
void ComputeCellIDKernel(const MpmParticleData pData,
                         int* outCellID, int numParticles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    float x = pData.pos_x[tid];
    float y = pData.pos_y[tid];

    int cx = (int)floorf(x / DX);
    int cy = (int)floorf(y / DY);
    if (cx < 0) cx = 0;
    if (cx >= NX) cx = NX - 1;
    if (cy < 0) cy = 0;
    if (cy >= NY) cy = NY - 1;

    outCellID[tid] = cy * NX + cx;
}

// ----------------------------------------------------------
// Kernel: Build cellStart / cellEnd from sorted cellID array
// ----------------------------------------------------------
__global__
void buildCellRanges(const int* __restrict__ cellIDSorted,
                     int* __restrict__ cellStart,
                     int* __restrict__ cellEnd,
                     int numParticles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    int cid = cellIDSorted[tid];

    if (tid == 0) {
        cellStart[cid] = 0;
    } else {
        int cidPrev = cellIDSorted[tid - 1];
        if (cid != cidPrev) {
            cellStart[cid] = tid;
            cellEnd[cidPrev] = tid;
        }
    }

    if (tid == numParticles - 1) {
        cellEnd[cid] = numParticles;
    }
}

// ----------------------------------------------------------
// P2G (Scatter) Kernel
// ----------------------------------------------------------
__global__
void P2GKernelScatter(MpmParticleData pData,
                      MpmGridNodeData gData,
                      const int* __restrict__ cellStart,
                      const int* __restrict__ cellEnd,
                      const int* __restrict__ particleIndices,
                      int numCells)
{
    int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNodes = (NX + 1) * (NY + 1);
    if (nodeID >= totalNodes) return;

    int nx_ = nodeID % (NX + 1);
    int ny_ = nodeID / (NX + 1);

    float massSum = 0.f;
    float momxSum = 0.f;
    float momySum = 0.f;

    // GIMP can extend 1 cell in each direction => 3x3
    int minCellX = nx_ - 1;
    int maxCellX = nx_ + 1;
    int minCellY = ny_ - 1;
    int maxCellY = ny_ + 1;

    if (minCellX < 0)       minCellX = 0;
    if (maxCellX >= NX)     maxCellX = NX - 1;
    if (minCellY < 0)       minCellY = 0;
    if (maxCellY >= NY)     maxCellY = NY - 1;

    for (int cy = minCellY; cy <= maxCellY; cy++) {
        for (int cx = minCellX; cx <= maxCellX; cx++) {
            int cellID = cy * NX + cx;
            int start = cellStart[cellID];
            int end   = cellEnd[cellID];

            for (int i = start; i < end; i++) {
                int p = particleIndices[i];

                float px  = pData.pos_x[p];
                float py  = pData.pos_y[p];
                float lpx = pData.halfSizeX[p];
                float lpy = pData.halfSizeY[p];

                float w = shapeGIMP2D(px, py, lpx, lpy,
                                      nx_, ny_, DX, DY);

                massSum += pData.mass[p] * w;
                momxSum += pData.mom_x[p] * w;
                momySum += pData.mom_y[p] * w;
            }
        }
    }

    gData.mass[nodeID]  = massSum;
    gData.mom_x[nodeID] = momxSum;
    gData.mom_y[nodeID] = momySum;
}

// ----------------------------------------------------------
// Kernel: Grid Update
// ----------------------------------------------------------
__global__
void GridUpdateKernel(MpmGridNodeData gData, int totalNodes)
{
    int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeID >= totalNodes) return;

    float m = gData.mass[nodeID];
    if (m > 1e-12f) {
        gData.vel_x[nodeID] = gData.mom_x[nodeID] / m;
        gData.vel_y[nodeID] = gData.mom_y[nodeID] / m;
    } else {
        gData.vel_x[nodeID] = 0.f;
        gData.vel_y[nodeID] = 0.f;
    }

    // update momentum
    gData.mom_x[nodeID] = gData.vel_x[nodeID] * m;
    gData.mom_y[nodeID] = gData.vel_y[nodeID] * m;
}

// ----------------------------------------------------------
// G2P (Gather) Kernel
// ----------------------------------------------------------
__global__
void G2PKernel(MpmParticleData pData,
               MpmGridNodeData gData,
               const int* __restrict__ cellID,
               int numParticles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    float px  = pData.pos_x[tid];
    float py  = pData.pos_y[tid];
    float lpx = pData.halfSizeX[tid];
    float lpy = pData.halfSizeY[tid];

    int cid = cellID[tid];
    int cx = cid % NX;
    int cy = cid / NX;

    // 3x3 node region
    int minNodeX = cx - 1;
    int maxNodeX = cx + 1;
    int minNodeY = cy - 1;
    int maxNodeY = cy + 1;

    if (minNodeX < 0)     minNodeX = 0;
    if (maxNodeX > NX)    maxNodeX = NX;
    if (minNodeY < 0)     minNodeY = 0;
    if (maxNodeY > NY)    maxNodeY = NY;

    float velxNew = 0.f;
    float velyNew = 0.f;
    float wsum    = 0.f;

    for (int ny_ = minNodeY; ny_ <= maxNodeY; ny_++) {
        for (int nx_ = minNodeX; nx_ <= maxNodeX; nx_++) {
            int nodeID = ny_ * (NX + 1) + nx_;
            float w = shapeGIMP2D(px, py, lpx, lpy,
                                  nx_, ny_, DX, DY);

            velxNew += gData.vel_x[nodeID] * w;
            velyNew += gData.vel_y[nodeID] * w;
            wsum    += w;
        }
    }

    if (wsum > 1e-12f) {
        velxNew /= wsum;
        velyNew /= wsum;
    }

    pData.vel_x[tid] = velxNew;
    pData.vel_y[tid] = velyNew;

    float m = pData.mass[tid];
    pData.mom_x[tid] = m * velxNew;
    pData.mom_y[tid] = m * velyNew;
}

// ----------------------------------------------------------
// Particle Update Kernel
// ----------------------------------------------------------
__global__
void ParticleUpdateKernel(MpmParticleData pData, int numParticles, float dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numParticles) return;

    pData.pos_x[tid] += pData.vel_x[tid] * dt;
    pData.pos_y[tid] += pData.vel_y[tid] * dt;
}
