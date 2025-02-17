#include "mpm_data.h"

//------------------------------------------------------------
// Allocate Particle Data on Device
//------------------------------------------------------------
MpmParticleData allocateParticleDataOnDevice(int numParticles)
{
    MpmParticleData p;
    size_t bytes = numParticles * sizeof(float);

    #define ALLOC_AND_CHECK(ptr)  CUDA_CHECK(cudaMalloc((void**)&(ptr), bytes))

    ALLOC_AND_CHECK(p.pos_x);
    ALLOC_AND_CHECK(p.pos_y);
    ALLOC_AND_CHECK(p.vel_x);
    ALLOC_AND_CHECK(p.vel_y);
    ALLOC_AND_CHECK(p.mass);
    ALLOC_AND_CHECK(p.mom_x);
    ALLOC_AND_CHECK(p.mom_y);

    ALLOC_AND_CHECK(p.halfSizeX);
    ALLOC_AND_CHECK(p.halfSizeY);

    ALLOC_AND_CHECK(p.stress_xx);
    ALLOC_AND_CHECK(p.stress_yy);
    ALLOC_AND_CHECK(p.stress_xy);
    ALLOC_AND_CHECK(p.strain_xx);
    ALLOC_AND_CHECK(p.strain_yy);
    ALLOC_AND_CHECK(p.strain_xy);
    ALLOC_AND_CHECK(p.grad_vx_x);
    ALLOC_AND_CHECK(p.grad_vx_y);
    ALLOC_AND_CHECK(p.grad_vy_x);
    ALLOC_AND_CHECK(p.grad_vy_y);

    return p;
}

//------------------------------------------------------------
// Free Particle Data
//------------------------------------------------------------
void freeParticleData(MpmParticleData p)
{
    CUDA_CHECK(cudaFree(p.pos_x));
    CUDA_CHECK(cudaFree(p.pos_y));
    CUDA_CHECK(cudaFree(p.vel_x));
    CUDA_CHECK(cudaFree(p.vel_y));
    CUDA_CHECK(cudaFree(p.mass));
    CUDA_CHECK(cudaFree(p.mom_x));
    CUDA_CHECK(cudaFree(p.mom_y));
    CUDA_CHECK(cudaFree(p.halfSizeX));
    CUDA_CHECK(cudaFree(p.halfSizeY));
    CUDA_CHECK(cudaFree(p.stress_xx));
    CUDA_CHECK(cudaFree(p.stress_yy));
    CUDA_CHECK(cudaFree(p.stress_xy));
    CUDA_CHECK(cudaFree(p.strain_xx));
    CUDA_CHECK(cudaFree(p.strain_yy));
    CUDA_CHECK(cudaFree(p.strain_xy));
    CUDA_CHECK(cudaFree(p.grad_vx_x));
    CUDA_CHECK(cudaFree(p.grad_vx_y));
    CUDA_CHECK(cudaFree(p.grad_vy_x));
    CUDA_CHECK(cudaFree(p.grad_vy_y));
}

//------------------------------------------------------------
// Allocate Grid Node Data on Device
//------------------------------------------------------------
MpmGridNodeData allocateGridNodeDataOnDevice(int numNodes)
{
    MpmGridNodeData g;
    size_t bytes = numNodes * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&g.mass,  bytes));
    CUDA_CHECK(cudaMalloc((void**)&g.mom_x, bytes));
    CUDA_CHECK(cudaMalloc((void**)&g.mom_y, bytes));
    CUDA_CHECK(cudaMalloc((void**)&g.vel_x, bytes));
    CUDA_CHECK(cudaMalloc((void**)&g.vel_y, bytes));

    return g;
}

//------------------------------------------------------------
// Free Grid Node Data
//------------------------------------------------------------
void freeGridNodeData(MpmGridNodeData g)
{
    CUDA_CHECK(cudaFree(g.mass));
    CUDA_CHECK(cudaFree(g.mom_x));
    CUDA_CHECK(cudaFree(g.mom_y));
    CUDA_CHECK(cudaFree(g.vel_x));
    CUDA_CHECK(cudaFree(g.vel_y));
}
