#pragma once

#include "../../nbodysolvers/gpu/nBodySolverGpuNaive.cuh"
#include "../../nbodysolvers/gpu/cudaHelper.cuh"

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"

#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <filesystem>




template <typename T>
__global__
void cudaTsneStepRep(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* labels, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* sumStorage, int tsneParticlesSize, float learnRate, float accelerationRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tsneParticlesSize)
    {
        sumStorage[i] = 0.0f;
        tsneParticles[i].derivative = glm::vec2(0.0f);

        for (int j = 0; j < tsneParticlesSize; j++)
        {
            if (i != j)
            {
                float softening = 1.0f;

                glm::vec2 diff = tsneParticles[j].position - tsneParticles[i].position;

                float distance = glm::length(diff);

                float oneOverDistance = 1.0f / (softening + (distance * distance));
                sumStorage[i] += 1.0f * oneOverDistance;

                tsneParticles[i].derivative += oneOverDistance * oneOverDistance * diff;
            }
        }

        //tsneParticles[i].derivative = glm::vec2(0.0f);
    }
}


template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tId)
{
    if (blockSize >= 64) sdata[tId] += sdata[tId + 32];
    if (blockSize >= 32) sdata[tId] += sdata[tId + 16];
    if (blockSize >= 16) sdata[tId] += sdata[tId + 8];
    if (blockSize >= 8) sdata[tId] += sdata[tId + 4];
    if (blockSize >= 4) sdata[tId] += sdata[tId + 2];
    if (blockSize >= 2) sdata[tId] += sdata[tId + 1];
}

template <typename T, unsigned int blockSize>
__global__
void cudaTsneStepSum(int sumInAmount, const float* sumIn, float* sumOut)
{
    extern __shared__ float sdata[];

    unsigned int tId = threadIdx.x;
    unsigned int inId = blockIdx.x * (blockSize * 2) + threadIdx.x; // blockSize * 2 because we add from current block and the next one
    unsigned int gridSize = blockSize * 2 * gridDim.x; // we have less blocks than we need so we can sum from what is not covered by them

    // load and add data into shared memory
    sdata[tId] = 0.0f;
    while (inId < sumInAmount)
    {
        sdata[tId] += ((inId < sumInAmount) ? sumIn[inId] : 0.0f) + ((inId + blockSize < sumInAmount) ? sumIn[inId + blockSize] : 0.0f);
        //sdata[tId] += sumIn[inId] + sumIn[inId + blockSize]; // can do this if array is power of blocksize * 2
        inId += gridSize;
    }

    __syncthreads();

    // take half the shared memory and let it sum the other half
    // unrolled for loop for every possible block size
    if (blockSize >= 1024)
        if (tId < 512) { sdata[tId] += sdata[tId + 512]; __syncthreads(); }

    if (blockSize >= 512)
        if (tId < 256) { sdata[tId] += sdata[tId + 256]; __syncthreads(); }

    if (blockSize >= 256)
        if (tId < 128) { sdata[tId] += sdata[tId + 128]; __syncthreads(); }

    if (blockSize >= 128)
        if (tId < 64) { sdata[tId] += sdata[tId + 64];   __syncthreads(); }

    // within a warp (which is 32) we dont have to sync up
    if (tId < 32) { warpReduce<blockSize>(sdata, tId); }

    // write shared memory result back to output
    if (tId == 0) { sumOut[blockIdx.x] = sdata[0]; }
}

template <typename T>
__global__
void cudaTsneStepAtt(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* labels, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* sumStorage, int tsneParticlesSize, float learnRate, float accelerationRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tsneParticlesSize)
    {
        tsneParticles[i].derivative *= -(1.0f / sumStorage[0]); // finish of the repulsive force

        // calculate attractive force
        for (int j = sparseMatrixColumnIndexStart[i]; j < sparseMatrixColumnIndexStart[i + 1]; j++)
        {
            int colIndex = indexTracker[i];
            int rowIndex = indexTracker[sparseMatrixCSC[j].row];

            glm::vec2 diff = tsneParticles[colIndex].position - tsneParticles[rowIndex].position;
            float distance = glm::length(diff);

            tsneParticles[colIndex].derivative += -(float)sparseMatrixCSC[j].val * (diff / (1.0f + (distance * distance)));
        }
        
        

        tsneParticles[i].position = tsneParticles[i].position + learnRate * tsneParticles[i].derivative;
        //tsneParticlesPrevPrev[i] = tsneParticlesPrev[i];
        //tsneParticlesPrev[i] = tsneParticles[i];

        //tsneParticles[indexTracker[i]].position = tsneParticlesPrev[indexTracker[i]].position + learnRate * tsneParticlesPrev[indexTracker[i]].derivative + accelerationRate * (tsneParticlesPrev[indexTracker[i]].position - tsneParticlesPrevPrev[indexTrackerPrev[i]].position);
        //tsneParticles[indexTracker[i]].label = tsneParticlesPrev[indexTracker[i]].label;
        //tsneParticles[indexTracker[i]].ID = tsneParticlesPrev[indexTracker[i]].ID;
    }
}


template <class T>
NBodySolverGpuNaive<T>::NBodySolverGpuNaive(int initTsneParticlesSize, SparseEntryCSC2D* initSparseMatrixCSC, size_t initSparseMatrixCSCSize, int* initSparseMatrixColumnIndexStart, int* initLabels, float initLearnRate, float initAccelerationRate)
{
    // initialize static memory
    tsneParticlesSize = initTsneParticlesSize;
    sparseMatrixCSCSize = initSparseMatrixCSCSize;
    learnRate = initLearnRate;
    accelerationRate = initAccelerationRate;

    // initialize dynamic memory on host
    int* indexTrackerToBuffer;
    int* indexTrackerPrevToBuffer;

    TsneParticle2D* tsneParticlesToBuffer;
    TsneParticle2D* tsneParticlesPrevToBuffer;
    TsneParticle2D* tsneParticlesPrevPrevToBuffer;

    cudaMallocHost((void**)&indexTrackerToBuffer,          tsneParticlesSize * sizeof(int));
    cudaMallocHost((void**)&indexTrackerPrevToBuffer,      tsneParticlesSize * sizeof(int));

    cudaMallocHost((void**)&tsneParticlesToBuffer,         tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMallocHost((void**)&tsneParticlesPrevToBuffer,     tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMallocHost((void**)&tsneParticlesPrevPrevToBuffer, tsneParticlesSize * sizeof(TsneParticle2D));

    // initialize dynamic memory on device
    cudaMalloc(&sparseMatrixCSC,              initSparseMatrixCSCSize * sizeof(SparseEntryCSC2D));
    cudaMalloc(&sparseMatrixColumnIndexStart, (tsneParticlesSize + 1) * sizeof(int));
    cudaMalloc(&labels,                       tsneParticlesSize * sizeof(int));

    cudaMalloc(&indexTracker,                 tsneParticlesSize * sizeof(int));
    cudaMalloc(&indexTrackerPrev,             tsneParticlesSize * sizeof(int));

    cudaMalloc(&tsneParticles,                tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMalloc(&tsneParticlesPrev,            tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMalloc(&tsneParticlesPrevPrev,        tsneParticlesSize * sizeof(TsneParticle2D));
        
    // create multiple sum storages
    {
        int reductionAmount = (SUMblockSize * 2) * SUMgridReductionAmount;

        // count how many different sum storages we need
        sumStorageAmount = 1;
        int amountTracker = tsneParticlesSize;
        while (amountTracker > 1)
        {
            sumStorageAmount++;
            amountTracker = divUp(amountTracker, reductionAmount);
        }

        // allocate space for them
        sumStorages = (float**)malloc(sumStorageAmount * sizeof(float*));
        sumStoragesAmounts = (int*)malloc(sumStorageAmount * sizeof(int));

        // create the actual device memory for the sum storage
        amountTracker = tsneParticlesSize;
        for (int i = 0; i < sumStorageAmount; i++)
        {
            checkCuda(cudaMalloc(&sumStorages[i], amountTracker * sizeof(float)));
            sumStoragesAmounts[i] = amountTracker;

            amountTracker = divUp(amountTracker, reductionAmount);
        }
    }

    // fill host memory with values
    srand(1952732);
    //srand(time(NULL));
    float sizeParam = 2.0f;
    for (int i = 0; i < tsneParticlesSize; i++)
    {
        float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        while (powf(randX, 2.0f) + powf(randY, 2.0f) > 1.0f)
        {
            randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        }

        glm::vec2 pos = glm::vec2(
            powf(sizeParam * randX, 1.0f),
            powf(sizeParam * randY, 1.0f)
        );

        int lab = initLabels[i];

        indexTrackerToBuffer[i] = i;
        indexTrackerPrevToBuffer[i] = i;

        tsneParticlesToBuffer[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab, i);
        tsneParticlesPrevToBuffer[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab, i);
        tsneParticlesPrevPrevToBuffer[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab, i);
    }

    // copy host to device
    cudaMemcpy(sparseMatrixCSC, initSparseMatrixCSC,                           initSparseMatrixCSCSize * sizeof(SparseEntryCSC2D), cudaMemcpyHostToDevice);
    cudaMemcpy(sparseMatrixColumnIndexStart, initSparseMatrixColumnIndexStart, (tsneParticlesSize + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(labels, initLabels,                                             tsneParticlesSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(indexTracker, indexTrackerToBuffer,                             tsneParticlesSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(indexTrackerPrev, indexTrackerPrevToBuffer,                     tsneParticlesSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(tsneParticles, tsneParticlesToBuffer,                           tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);
    cudaMemcpy(tsneParticlesPrev, tsneParticlesPrevToBuffer,                   tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);
    cudaMemcpy(tsneParticlesPrevPrev, tsneParticlesPrevPrevToBuffer,           tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);

    // free host memory
    delete[] initSparseMatrixCSC;
    delete[] initSparseMatrixColumnIndexStart;
    delete[] initLabels;

    cudaFreeHost(indexTrackerToBuffer);
    cudaFreeHost(indexTrackerPrevToBuffer);

    cudaFreeHost(tsneParticlesToBuffer);
    cudaFreeHost(tsneParticlesPrevToBuffer);
    cudaFreeHost(tsneParticlesPrevPrevToBuffer);
}

template <class T>
NBodySolverGpuNaive<T>::~NBodySolverGpuNaive()
{
    cudaFree(sparseMatrixCSC);
    cudaFree(sparseMatrixColumnIndexStart);
    cudaFree(labels);

    cudaFree(indexTracker);
    cudaFree(indexTrackerPrev);

    cudaFree(tsneParticles);
    cudaFree(tsneParticlesPrev);
    cudaFree(tsneParticlesPrevPrev);

    for (int i = 0; i < sumStorageAmount; i++)
        checkCuda(cudaFree(sumStorages[i]));
    free(sumStorages);
    free(sumStoragesAmounts);
}

template <class T>
void NBodySolverGpuNaive<T>::timeStep()
{
    int blockSize = 256;
    int numBlocks = divUp(tsneParticlesSize, blockSize);

    for (int r = 0; r < 1; r++)
    {
        cudaTsneStepRep<TsneParticle2D> << <numBlocks, blockSize >> > (sparseMatrixCSC, sparseMatrixCSCSize, sparseMatrixColumnIndexStart, labels, indexTracker, indexTrackerPrev, tsneParticles, tsneParticlesPrev, tsneParticlesPrevPrev, sumStorages[0], tsneParticlesSize, learnRate, accelerationRate);

        for (int i = 0; i < sumStorageAmount - 1; i++)
        {
            int sumNumBlocks = divUp(sumStoragesAmounts[i], (SUMblockSize * 2) * SUMgridReductionAmount);
            cudaTsneStepSum<TsneParticle2D, 128> << <sumNumBlocks, SUMblockSize, SUMblockSize * sizeof(float) + 1 >> > (sumStoragesAmounts[i], sumStorages[i], sumStorages[i + 1]);
        }

        cudaTsneStepAtt<TsneParticle2D> << <numBlocks, blockSize >> > (sparseMatrixCSC, sparseMatrixCSCSize, sparseMatrixColumnIndexStart, labels, indexTracker, indexTrackerPrev, tsneParticles, tsneParticlesPrev, tsneParticlesPrevPrev, sumStorages[sumStorageAmount - 1], tsneParticlesSize, learnRate, accelerationRate);
    }
}

template <class T>
void NBodySolverGpuNaive<T>::getParticles(std::vector<TsneParticle2D>& result)
{
    cudaMemcpy(result.data(), tsneParticles, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyDeviceToHost);
}


// Explicit instantiation (required for templates in .cu)
template class NBodySolverGpuNaive<TsneParticle2D>;
template __global__ void cudaTsneStepRep<TsneParticle2D>(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* labels, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* sumStorage, int tsneParticlesSize, float learnRate, float accelerationRate);
template __global__ void cudaTsneStepSum<TsneParticle2D, 128>(int sumInAmount, const float* sumIn, float* sumOut);
template __global__ void cudaTsneStepAtt<TsneParticle2D>(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* labels, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* sumStorage, int tsneParticlesSize, float learnRate, float accelerationRate);