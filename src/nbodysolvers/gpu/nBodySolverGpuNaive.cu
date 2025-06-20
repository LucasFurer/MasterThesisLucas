#include <iostream>
#include "../../nbodysolvers/gpu/nBodySolverGpuNaive.cuh"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>



//template <typename T>
__global__ 
void cudaComputeForcesKernel(float* accumulator, GpuTsneParticle2D* particles, int* indexTracker, int N)
{
    float localAccumulator = 0.0f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        //particles[indexTracker[i]].derivative = make_float2(0.0f,0.0f);
        //*accumulator = 1.0f;

        for (int j = 0; j < N; j++)
        {
            if (i != j)
            {
                GpuTsneParticle2D passiveParticle = particles[indexTracker[i]];
                GpuTsneParticle2D activeParticle = particles[indexTracker[j]];

                float softening = 1.0f;

                float diffX = activeParticle.position.x - passiveParticle.position.x;
                float diffY = activeParticle.position.y - passiveParticle.position.y;
                float distance = sqrt(diffX * diffX + diffY * diffY);//glm::length(diff);

                float oneOverDistance = 1.0f / (softening + (distance * distance));
                localAccumulator += oneOverDistance;
                //atomicAdd(accumulator, oneOverDistance);
                //accumulator += 1.0f * oneOverDistance;
                //*accumulator = 1.0f;

                passiveParticle.derivative.x += oneOverDistance* oneOverDistance* diffX;
                passiveParticle.derivative.y += oneOverDistance* oneOverDistance* diffY;

                particles[indexTracker[i]].derivative.x = passiveParticle.derivative.x;
                particles[indexTracker[i]].derivative.y = passiveParticle.derivative.y;

            }
        }
    }


    atomicAdd(accumulator, localAccumulator);
}

template <typename T>
void cudaComputeForcesNaive(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker)
{
    int N = particles.size();

    // Allocate device memory
    float* d_accumulator;
    GpuTsneParticle2D* d_particles;
    int* d_indexTracker;

    cudaMalloc(&d_accumulator,  sizeof(float));
    cudaMalloc(&d_particles,    N * sizeof(T));
    cudaMalloc(&d_indexTracker, N * sizeof(int));

    std::vector<GpuTsneParticle2D> gpuParticles(N);
    for (int i = 0; i < N; i++) 
        gpuParticles[i] = GpuTsneParticle2D(particles[i].position, particles[i].derivative, particles[i].label, particles[i].ID);
    

    // Copy from host to device
    cudaMemcpy(d_accumulator,  &accumulator,        sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles,    gpuParticles.data(), N * sizeof(GpuTsneParticle2D),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_indexTracker, indexTracker.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    //cudaOccupancyMaxPotentialBlockSize(&blockSize, &numBlocks, cudaComputeForcesKernel);

    //int minGridSize = 0;
    //int blockSize = 0;
    //cudaOccupancyMaxPotentialBlockSize(
    //    &minGridSize,       // Output: minimum number of blocks to achieve max occupancy
    //    &blockSize,         // Output: block size to use
    //    cudaComputeForcesKernel    // Your kernel
    //);
    //int gridSize = (N + blockSize - 1) / blockSize;
    ////myKernelFunction << <gridSize, blockSize >> > (...);


    // Kernel launch
    //std::cout << "calling kernel" << std::endl;
    cudaComputeForcesKernel<<<numBlocks, blockSize>>>(d_accumulator, d_particles, d_indexTracker, N);
    //cudaComputeForcesKernel<<<gridSize, blockSize>>>(d_accumulator, d_particles, d_indexTracker, N);

    // Copy result back to host
    cudaMemcpy(&accumulator,        d_accumulator,  sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuParticles.data(), d_particles,    N * sizeof(GpuTsneParticle2D),   cudaMemcpyDeviceToHost);
    cudaMemcpy(indexTracker.data(), d_indexTracker, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        particles[i].derivative = glm::vec2(gpuParticles[i].derivative.x, gpuParticles[i].derivative.y);
        //std::cout << "gpuParticles outputX: " << gpuParticles[i].derivative.x << " gpuParticles outputY: " << gpuParticles[i].derivative.y << std::endl;
        //std::cout << "particles outputX: " << particles[i].derivative.x << " particles outputY: " << particles[i].derivative.y << std::endl;
    }

    // copy results parameters
    //accumulator = *d_accumulator;
    //particles.assign(particles, particles + N);
    //indexTracker.assign(indexTracker, indexTracker + N);

    // Free device memory
    cudaFree(d_accumulator);
    cudaFree(d_particles);
    cudaFree(d_indexTracker);
}

template void cudaComputeForcesNaive<TsneParticle2D>(float& accumulator, std::vector<TsneParticle2D>& particles, std::vector<int>& indexTracker);