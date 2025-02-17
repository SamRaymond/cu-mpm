#include "mpm_common.h"
#include "mpm_data.h"
#include "mpm_kernels.cuh"
#include "mpm_io.h"

int main()
{
    // 1. Allocate particle data on device
    MpmParticleData pData = allocateParticleDataOnDevice(NUM_PARTICLES);

    // 2. Initialize particles
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
        InitParticlesKernel<<<grid, block>>>(pData, NUM_PARTICLES,
                                             /*vxInit=*/1.0f, /*vyInit=*/0.0f);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Prepare for sorting & building cellStart/cellEnd
    thrust::device_vector<int> d_cellID(NUM_PARTICLES);
    thrust::device_vector<int> d_particleIndices(NUM_PARTICLES);
    thrust::sequence(d_particleIndices.begin(), d_particleIndices.end());

    int numCells = NX * NY;
    thrust::device_vector<int> d_cellStart(numCells, 0);
    thrust::device_vector<int> d_cellEnd(numCells, 0);

    // 3. Allocate grid data
    int totalNodes = (NX + 1) * (NY + 1);
    MpmGridNodeData gData = allocateGridNodeDataOnDevice(totalNodes);

    // Host arrays for saving .vtp
    float* hostPosX = (float*)malloc(NUM_PARTICLES * sizeof(float));
    float* hostPosY = (float*)malloc(NUM_PARTICLES * sizeof(float));
    float* hostVelX = (float*)malloc(NUM_PARTICLES * sizeof(float));
    float* hostVelY = (float*)malloc(NUM_PARTICLES * sizeof(float));

    // 4. Main loop
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        // Write .vtp every SAVE_INTERVAL steps
        if (iter % SAVE_INTERVAL == 0)
        {
            CUDA_CHECK(cudaMemcpy(hostPosX, pData.pos_x,
                                  NUM_PARTICLES * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostPosY, pData.pos_y,
                                  NUM_PARTICLES * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostVelX, pData.vel_x,
                                  NUM_PARTICLES * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hostVelY, pData.vel_y,
                                  NUM_PARTICLES * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            char fname[256];
            sprintf(fname, "particles_%06d.vtp", iter);
            std::cout << "Writing " << fname << std::endl;
            SaveParticleDataToVTP(fname, hostPosX, hostPosY,
                                  hostVelX, hostVelY, NUM_PARTICLES);
        }

        // Compute cellID
        {
            dim3 block(BLOCK_SIZE);
            dim3 gridDim((NUM_PARTICLES + block.x - 1) / block.x);
            ComputeCellIDKernel<<<gridDim, block>>>(pData,
                                                    thrust::raw_pointer_cast(d_cellID.data()),
                                                    NUM_PARTICLES);
        }

        // Sort by cellID (uncomment if desired to keep particles in ascending cell order)
        // thrust::sort_by_key(d_cellID.begin(), d_cellID.end(), d_particleIndices.begin());

        // Build cellStart / cellEnd
        {
            thrust::fill(d_cellStart.begin(), d_cellStart.end(), 0);
            thrust::fill(d_cellEnd.begin(),   d_cellEnd.end(),   0);

            dim3 block(BLOCK_SIZE);
            dim3 gridDim((NUM_PARTICLES + block.x - 1) / block.x);
            buildCellRanges<<<gridDim, block>>>(
                thrust::raw_pointer_cast(d_cellID.data()),
                thrust::raw_pointer_cast(d_cellStart.data()),
                thrust::raw_pointer_cast(d_cellEnd.data()),
                NUM_PARTICLES);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Reset grid
        CUDA_CHECK(cudaMemset(gData.mass,  0, totalNodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(gData.mom_x, 0, totalNodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(gData.mom_y, 0, totalNodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(gData.vel_x, 0, totalNodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(gData.vel_y, 0, totalNodes * sizeof(float)));

        // P2G scatter
        {
            dim3 block(BLOCK_SIZE);
            dim3 gridDim((totalNodes + block.x - 1) / block.x);
            P2GKernelScatter<<<gridDim, block>>>(
                pData, gData,
                thrust::raw_pointer_cast(d_cellStart.data()),
                thrust::raw_pointer_cast(d_cellEnd.data()),
                thrust::raw_pointer_cast(d_particleIndices.data()),
                numCells
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Grid update
        {
            dim3 block(BLOCK_SIZE);
            dim3 gridDim((totalNodes + block.x - 1) / block.x);
            GridUpdateKernel<<<gridDim, block>>>(gData, totalNodes);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // G2P gather
        {
            dim3 block(BLOCK_SIZE);
            dim3 gridDim((NUM_PARTICLES + block.x - 1) / block.x);
            G2PKernel<<<gridDim, block>>>(pData, gData,
                                          thrust::raw_pointer_cast(d_cellID.data()),
                                          NUM_PARTICLES);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Particle update
        {
            dim3 block(BLOCK_SIZE);
            dim3 gridDim((NUM_PARTICLES + block.x - 1) / block.x);
            ParticleUpdateKernel<<<gridDim, block>>>(pData, NUM_PARTICLES, DT);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Optional progress print
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << "/" << MAX_ITER << std::endl;
        }
    }

    // Cleanup
    freeParticleData(pData);
    freeGridNodeData(gData);
    free(hostPosX);
    free(hostPosY);
    free(hostVelX);
    free(hostVelY);

    std::cout << "Done.\n";
    return 0;
}
