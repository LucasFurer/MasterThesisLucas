#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cstdint> // for uint8_t
#include <string>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <cstddef>

//#define GLM_ENABLE_EXPERIMENTAL
//#define GLM_FORCE_CUDA
//#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "nBodySolverGpuBH.cuh"
#include "cudaHelper.cuh"
#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"
#include "../../trees/gpu/quadTreeGpuBH.cuh"
#include "../../common.h"


// reduction functions -----------------------------------------------------------------------------------------------------------

struct SumOp 
{
    __device__ float operator()(float a, float b) const 
    {
        return a + b;
    }
};

struct MinOp 
{
    __device__ float operator()(float a, float b) const 
    {
        return std::fminf(a, b);
    }
};

struct MaxOp 
{
    __device__ float operator()(float a, float b) const 
    {
        return std::fmaxf(a, b);
    }
};


// repulsive ----------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void fillReductionX(TsneParticle2D* tsneParticles, float* toFill, int tsneParticlesSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tsneParticlesSize)
    {
        toFill[i] = tsneParticles[i].position.x;
    }
}
template <typename T>
__global__
void fillReductionY(TsneParticle2D* tsneParticles, float* toFill, int tsneParticlesSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tsneParticlesSize)
    {
        toFill[i] = tsneParticles[i].position.y;
    }
}

template <typename T>
__global__
void createRoot(int tsneParticlesSize, float* xMin, float* yMin, float* xMax, float* yMax, QuadTreeGpuBH<TsneParticle2D>* nodes)
{
    float lengthX = *xMax - *xMin;
    float lengthY = *yMax - *yMin;
    //
    //float largestAxis = std::fmaxf(lengthX, lengthY);

    //glm::vec2 BBcentre = glm::vec2(*xMin + largestAxis * 0.5f, *yMin + largestAxis * 0.5f);

    float largestAxis = std::fmaxf(lengthX, lengthY);
    glm::vec2 BBcentre = glm::vec2((*xMin + *xMax) * 0.5f, (*yMin + *yMax) * 0.5f);


    nodes[0] = QuadTreeGpuBH<TsneParticle2D>(BBcentre, largestAxis, 0, 0, tsneParticlesSize, 0.0f, glm::vec2(0.0f));
}

template <typename T>
__global__
void createChildren(int tsneParticlesSize, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesBuffer, int* indexTracker, QuadTreeGpuBH<TsneParticle2D>* nodes, int nodesSize, int levelToDivide, int numberOfNodesInCurrentLevel, int startIndexFirstNodeInCurrentLevel, int startIndexFirstNodeInNextLevel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int numberOfNodes = numberOfNodesInCurrentLevel;

    if (i < numberOfNodes)
    {
        QuadTreeGpuBH<TsneParticle2D> currentNode = nodes[startIndexFirstNodeInCurrentLevel + i];

        // sort x
        int xLow = currentNode.firstParticleIndex;
        int xHigh = currentNode.firstParticleIndex + currentNode.particleIndexAmount - 1;
        for (int currentNodeParticleIndex = currentNode.firstParticleIndex; currentNodeParticleIndex < currentNode.firstParticleIndex + currentNode.particleIndexAmount; currentNodeParticleIndex++)
        {
            TsneParticle2D particle = tsneParticles[currentNodeParticleIndex];

            if (particle.position.x <= currentNode.BBcentre.x)
            {
                tsneParticlesBuffer[xLow] = particle;
                xLow++;
            }
            else
            {
                tsneParticlesBuffer[xHigh] = particle;
                xHigh--;
            }
        }
        int xFirstIndexHigh = xHigh + 1;

        // sort y of lower x
        int yLowA = currentNode.firstParticleIndex;
        int yHighA = xFirstIndexHigh - 1;
        for (int currentNodeParticleIndex = currentNode.firstParticleIndex; currentNodeParticleIndex < xFirstIndexHigh; currentNodeParticleIndex++)
        {
            TsneParticle2D particle = tsneParticlesBuffer[currentNodeParticleIndex];

            if (particle.position.y <= currentNode.BBcentre.y)
            {
                tsneParticles[yLowA] = particle;
                indexTracker[particle.ID] = yLowA;
                yLowA++;
            }
            else
            {
                tsneParticles[yHighA] = particle;
                indexTracker[particle.ID] = yHighA;
                yHighA--;
            }
        }
        int yFirstIndexHighA = yHighA + 1;

        // sort y of higher x
        int yLowB = xFirstIndexHigh;
        int yHighB = currentNode.firstParticleIndex + currentNode.particleIndexAmount - 1;
        for (int currentNodeParticleIndex = xFirstIndexHigh; currentNodeParticleIndex < currentNode.firstParticleIndex + currentNode.particleIndexAmount; currentNodeParticleIndex++)
        {
            TsneParticle2D particle = tsneParticlesBuffer[currentNodeParticleIndex];

            if (particle.position.y <= currentNode.BBcentre.y)
            {
                tsneParticles[yLowB] = particle;
                indexTracker[particle.ID] = yLowB;
                yLowB++;
            }
            else
            {
                tsneParticles[yHighB] = particle;
                indexTracker[particle.ID] = yHighB;
                yHighB--;
            }
        }
        int yFirstIndexHighB = yHighB + 1;

        // set children parameters
        float BBlengthABCD = 0.5f * currentNode.BBlength;

        glm::vec2 BBcentreA = currentNode.BBcentre + glm::vec2(-0.25f * currentNode.BBlength, -0.25f * currentNode.BBlength);
        unsigned int firstChildIndexA = 0;
        unsigned int firstParticleIndexA = currentNode.firstParticleIndex;
        unsigned int particleIndexAmountA = yFirstIndexHighA - firstParticleIndexA;

        glm::vec2 BBcentreB = currentNode.BBcentre + glm::vec2(-0.25f * currentNode.BBlength, 0.25f * currentNode.BBlength);
        unsigned int firstChildIndexB = 0;
        unsigned int firstParticleIndexB = yFirstIndexHighA;
        unsigned int particleIndexAmountB = xFirstIndexHigh - yFirstIndexHighA;

        glm::vec2 BBcentreC = currentNode.BBcentre + glm::vec2(0.25f * currentNode.BBlength, -0.25f * currentNode.BBlength);
        unsigned int firstChildIndexC = 0;
        unsigned int firstParticleIndexC = xFirstIndexHigh;
        unsigned int particleIndexAmountC = yFirstIndexHighB - xFirstIndexHigh;

        glm::vec2 BBcentreD = currentNode.BBcentre + glm::vec2(0.25f * currentNode.BBlength, 0.25f * currentNode.BBlength);
        unsigned int firstChildIndexD = 0;
        unsigned int firstParticleIndexD = yFirstIndexHighB;
        unsigned int particleIndexAmountD = (currentNode.firstParticleIndex + currentNode.particleIndexAmount) - yFirstIndexHighB;

        // create children
        nodes[startIndexFirstNodeInCurrentLevel + i].firstChildIndex = startIndexFirstNodeInNextLevel + (i * 4) + 0;
        nodes[startIndexFirstNodeInNextLevel + (i * 4) + 0] = QuadTreeGpuBH<TsneParticle2D>(BBcentreA, BBlengthABCD, firstChildIndexA, firstParticleIndexA, particleIndexAmountA, 0, glm::vec2(0.0f));
        nodes[startIndexFirstNodeInNextLevel + (i * 4) + 1] = QuadTreeGpuBH<TsneParticle2D>(BBcentreB, BBlengthABCD, firstChildIndexB, firstParticleIndexB, particleIndexAmountB, 0, glm::vec2(0.0f));
        nodes[startIndexFirstNodeInNextLevel + (i * 4) + 2] = QuadTreeGpuBH<TsneParticle2D>(BBcentreC, BBlengthABCD, firstChildIndexC, firstParticleIndexC, particleIndexAmountC, 0, glm::vec2(0.0f));
        nodes[startIndexFirstNodeInNextLevel + (i * 4) + 3] = QuadTreeGpuBH<TsneParticle2D>(BBcentreD, BBlengthABCD, firstChildIndexD, firstParticleIndexD, particleIndexAmountD, 0, glm::vec2(0.0f));
    }
}

template <typename T>
__global__
void cudaSumLeavesBH(TsneParticle2D* tsneParticles, QuadTreeGpuBH<TsneParticle2D>* nodes, int numberOfNodesInCurrentLevel, int startIndexFirstNodeInCurrentLevel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int numberOfNodes = numberOfNodesInCurrentLevel;

    if (i < numberOfNodes)
    {
        QuadTreeGpuBH<TsneParticle2D> currentNode = nodes[startIndexFirstNodeInCurrentLevel + i];

        float totalMass = 0.0f;
        glm::vec2 centreOfMass = glm::vec2(0.0f);

        for (int i = currentNode.firstParticleIndex; i < currentNode.firstParticleIndex + currentNode.particleIndexAmount; i++)
        {
            totalMass += 1.0f; //tsneParticles[i].mass;
            centreOfMass += tsneParticles[i].position; // tsneParticles[i].position * tsneParticles[i].mass;
        }

        if (totalMass != 0.0f)
            centreOfMass /= totalMass;
        nodes[startIndexFirstNodeInCurrentLevel + i].totalMass = totalMass;
        nodes[startIndexFirstNodeInCurrentLevel + i].centreOfMass = centreOfMass;
    }
}

template <typename T>
__global__
void cudaSumNodesBH(QuadTreeGpuBH<TsneParticle2D>* nodes, int nodesSize, int numberOfNodesInCurrentLevel, int startIndexFirstNodeInCurrentLevel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int numberOfNodes = numberOfNodesInCurrentLevel;

    if (i < numberOfNodes)
    {
        QuadTreeGpuBH<TsneParticle2D> currentNode = nodes[startIndexFirstNodeInCurrentLevel + i];

        float totalMass = 0.0f;
        glm::vec2 centreOfMass = glm::vec2(0.0f);

        for (int i = currentNode.firstChildIndex; i < currentNode.firstChildIndex + 4; i++)
        {
            totalMass += nodes[i].totalMass;
            centreOfMass += nodes[i].centreOfMass * nodes[i].totalMass; // tsneParticles[i].position * tsneParticles[i].mass;
        }

        if (totalMass != 0.0f)
            centreOfMass /= totalMass;
        nodes[startIndexFirstNodeInCurrentLevel + i].totalMass = totalMass;
        nodes[startIndexFirstNodeInCurrentLevel + i].centreOfMass = centreOfMass;
    }
}

template <typename T>
__global__
void cudaTsneBHStepRep(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, QuadTreeGpuBH<TsneParticle2D>* nodes, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < tsneParticlesSize)
    {
        reductionStorage[i] = 0.0f;
        tsneParticles[i].derivative = glm::vec2(0.0f);

        const int treeDepth = 15;
        QuadTreeGpuBH<TsneParticle2D> stack[treeDepth *4];
        int stackPointer = 0;
        stack[0] = nodes[0];

        while (stackPointer != -1)
        {
            QuadTreeGpuBH<TsneParticle2D> currentNode = stack[stackPointer];
            stackPointer--;

            glm::vec2 nodeDiff = tsneParticles[i].position - currentNode.centreOfMass;
            float nodeDiffLength = std::sqrtf(nodeDiff.x * nodeDiff.x + nodeDiff.y * nodeDiff.y);

            if (currentNode.BBlength / nodeDiffLength < 1.0f)
            {
                float softening = 1.0f;

                glm::vec2 diff = currentNode.centreOfMass - tsneParticles[i].position;

                //float distance = glm::length(diff);
                float distance = std::sqrtf(diff.x * diff.x + diff.y * diff.y);

                float oneOverDistance = 1.0f / (softening + (distance * distance));
                reductionStorage[i] += currentNode.totalMass * oneOverDistance;

                tsneParticles[i].derivative += currentNode.totalMass * oneOverDistance * oneOverDistance * diff;
            }
            else if (currentNode.firstChildIndex == 0) // if you have zero children (or if you have only 1 TODO)
            {
                for (int j = currentNode.firstParticleIndex; j < currentNode.firstParticleIndex + currentNode.particleIndexAmount; j++)
                {
                    if (!glm::all(glm::equal(tsneParticles[j].position, tsneParticles[i].position)))
                    {
                        float softening = 1.0f;

                        glm::vec2 diff = tsneParticles[j].position - tsneParticles[i].position;

                        //float distance = glm::length(diff);
                        float distance = std::sqrtf(diff.x * diff.x + diff.y * diff.y);

                        float oneOverDistance = 1.0f / (softening + (distance * distance));
                        reductionStorage[i] += 1.0f * oneOverDistance;

                        tsneParticles[i].derivative += oneOverDistance * oneOverDistance * diff;
                    }
                }
            }
            else
            {
                for (int n = currentNode.firstChildIndex; n < currentNode.firstChildIndex + 4; n++)
                {
                    stackPointer++;
                    stack[stackPointer] = nodes[n];
                }
            }
        }



        //for (int j = 0; j < tsneParticlesSize; j++)
        //{
        //    if (i != j)
        //    {
        //        //int indexJ = indexTracker[j];

        //        float softening = 1.0f;

        //        glm::vec2 diff = tsneParticles[j].position - tsneParticles[i].position;

        //        float distance = glm::length(diff);

        //        float oneOverDistance = 1.0f / (softening + (distance * distance));
        //        reductionStorage[i] += 1.0f * oneOverDistance;

        //        tsneParticles[i].derivative += oneOverDistance * oneOverDistance * diff;
        //    }
        //}
    }
}


template <typename Op, unsigned int blockSize>
__device__ void warpReduceBH(volatile float* sdata, int tId, Op op)
{
    if (blockSize >= 64) sdata[tId] = op(sdata[tId], sdata[tId + 32]);
    if (blockSize >= 32) sdata[tId] = op(sdata[tId], sdata[tId + 16]);
    if (blockSize >= 16) sdata[tId] = op(sdata[tId], sdata[tId + 8]);
    if (blockSize >= 8)  sdata[tId] = op(sdata[tId], sdata[tId + 4]);
    if (blockSize >= 4)  sdata[tId] = op(sdata[tId], sdata[tId + 2]);
    if (blockSize >= 2)  sdata[tId] = op(sdata[tId], sdata[tId + 1]);
}

template <typename Op, unsigned int blockSize>
__global__
void cudaTsneReduceBH(int inAmount, const float* in, float* out, Op op, float identity)
{
    extern __shared__ float sdata[];

    unsigned int tId = threadIdx.x;
    unsigned int inId = blockIdx.x * (blockSize * 2) + threadIdx.x; // blockSize * 2 because we add from current block and the next one
    unsigned int gridSize = blockSize * 2 * gridDim.x; // we have less blocks than we need so we can sum from what is not covered by them

    // load and add data into shared memory
    sdata[tId] = identity;
    while (inId < inAmount)
    {
        float a = (inId < inAmount) ? in[inId] : identity;
        float b = (inId + blockSize < inAmount) ? in[inId + blockSize] : identity;

        sdata[tId] = op(sdata[tId], op(a, b));
        //sdata[tId] += sumIn[inId] + sumIn[inId + blockSize]; // can do this if array is power of blocksize * 2
        inId += gridSize;
    }

    __syncthreads();

    // take half the shared memory and let it sum the other half
    // unrolled for loop for every possible block size
    if (blockSize >= 1024)
        if (tId < 512) { sdata[tId] = op(sdata[tId], sdata[tId + 512]); __syncthreads(); }

    if (blockSize >= 512)
        if (tId < 256) { sdata[tId] = op(sdata[tId], sdata[tId + 256]); __syncthreads(); }

    if (blockSize >= 256)
        if (tId < 128) { sdata[tId] = op(sdata[tId], sdata[tId + 128]); __syncthreads(); }

    if (blockSize >= 128)
        if (tId < 64) { sdata[tId] = op(sdata[tId], sdata[tId + 64]);   __syncthreads(); }


    // within a warp (which is 32) we dont have to sync up
    if (tId < 32) { warpReduceBH<Op, blockSize>(sdata, tId, op); }

    // write shared memory result back to output
    if (tId == 0) { out[blockIdx.x] = sdata[0]; }
}

// attractive ----------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void cudaTsneBHStepAtt(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < tsneParticlesSize)
    {
        // get index trackers
        int indexTrackerResult = indexTracker[i];
        int indexTrackerPrevResult = indexTrackerPrev[i];

        // finish of the repulsive force
        tsneParticles[indexTrackerResult].derivative *= -(1.0f / reductionStorage[0]);

        // calculate attractive force
        for (int j = sparseMatrixColumnIndexStart[i]; j < sparseMatrixColumnIndexStart[i + 1]; j++)
        {
            int rowIndex = indexTracker[sparseMatrixCSC[j].row];

            glm::vec2 diff = tsneParticles[indexTrackerResult].position - tsneParticles[rowIndex].position;
            float distance = glm::length(diff);

            tsneParticles[indexTrackerResult].derivative += -(float)sparseMatrixCSC[j].val * (diff / (1.0f + (distance * distance)));
        }
    }
}

// update ----------------------------------------------------------------------------------------------------------------------

template <typename T>
__global__
void cudaTsneBHStepUpd(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < tsneParticlesSize)
    {
        // get index trackers
        int indexTrackerResult = indexTracker[i];
        int indexTrackerPrevResult = indexTrackerPrev[i];

        // apply the derivative to new position
        tsneParticles[indexTrackerResult].position = tsneParticlesPrev[indexTrackerResult].position + learnRate * tsneParticlesPrev[indexTrackerResult].derivative + accelerationRate * (tsneParticlesPrev[indexTrackerResult].position - tsneParticlesPrevPrev[indexTrackerPrevResult].position);
        tsneParticles[indexTrackerResult].label = tsneParticlesPrev[indexTrackerResult].label;
        tsneParticles[indexTrackerResult].ID = tsneParticlesPrev[indexTrackerResult].ID;

        // copy indices to the prev
        indexTrackerPrev[i] = indexTracker[i];
    }
}

// class ----------------------------------------------------------------------------------------------------------------------

template <class T>
NBodySolverGpuBH<T>::NBodySolverGpuBH(int initTsneParticlesSize, SparseEntryCSC2D* initSparseMatrixCSC, size_t initSparseMatrixCSCSize, int* initSparseMatrixColumnIndexStart, std::vector<uint8_t>& initLabels, float initLearnRate, float initAccelerationRate, int initTreeDepth)
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

    cudaMallocHost((void**)&indexTrackerToBuffer, tsneParticlesSize * sizeof(int));
    cudaMallocHost((void**)&indexTrackerPrevToBuffer, tsneParticlesSize * sizeof(int));

    cudaMallocHost((void**)&tsneParticlesToBuffer, tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMallocHost((void**)&tsneParticlesPrevToBuffer, tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMallocHost((void**)&tsneParticlesPrevPrevToBuffer, tsneParticlesSize * sizeof(TsneParticle2D));

    // initialize dynamic memory on device
    cudaMalloc(&sparseMatrixCSC, initSparseMatrixCSCSize * sizeof(SparseEntryCSC2D));
    cudaMalloc(&sparseMatrixColumnIndexStart, (tsneParticlesSize + 1) * sizeof(int));
    cudaMalloc(&labels, tsneParticlesSize * sizeof(int));

    cudaMalloc(&indexTracker, tsneParticlesSize * sizeof(int));
    cudaMalloc(&indexTrackerPrev, tsneParticlesSize * sizeof(int));

    cudaMalloc(&tsneParticles, tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMalloc(&tsneParticlesPrev, tsneParticlesSize * sizeof(TsneParticle2D));
    cudaMalloc(&tsneParticlesPrevPrev, tsneParticlesSize * sizeof(TsneParticle2D));

    treeDepth = initTreeDepth;
    depthSizes = new int[treeDepth + 1];
    depthStarts = new int[treeDepth + 1];
    depthStarts[0] = 0;
    nodesSize = 0;
    for (int i = 0; i <= treeDepth; i++) // treeDepth = 0 mean just the root
    {
        int currentLevelSize = 1;
        for (int j = 0; j < i; j++)
        {
            currentLevelSize *= 4;
        }
        depthSizes[i] = currentLevelSize;
        nodesSize += currentLevelSize; // we split into 4 new children each depth so 4 to the power of levelDepth

        int currentDepthStart = 0;
        for (int j = 0; j <= i; j++)
        {
            //std::cout << "j: " << j << ", i: " << i << std::endl;
            //std::cout << "depthSizes[j]: " << depthSizes[j] << std::endl;
            currentDepthStart += depthSizes[j];
        }
        if (i + 1 < treeDepth + 1)
            depthStarts[i + 1] = currentDepthStart;
    }

    cudaMalloc(&nodes, nodesSize * sizeof(QuadTreeGpuBH<TsneParticle2D>));

    cudaMalloc(&xMin, 1 * sizeof(float));
    cudaMalloc(&yMin, 1 * sizeof(float));
    cudaMalloc(&xMax, 1 * sizeof(float));
    cudaMalloc(&yMax, 1 * sizeof(float));
    cudaMalloc(&repulsionSum, 1 * sizeof(float));

    // create multiple reduction storages
    {
        int reductionAmount = (reductionBlockSize * 2) * reductionGridReductionAmount;

        // count how many different reduction storages we need
        reductionStorageAmount = 0;
        int amountTracker = tsneParticlesSize;
        while (amountTracker > 1)
        {
            reductionStorageAmount++;
            amountTracker = divUp(amountTracker, reductionAmount);
        }

        // allocate space for them
        reductionStorages = (float**)malloc(reductionStorageAmount * sizeof(float*));
        reductionStoragesAmounts = (int*)malloc(reductionStorageAmount * sizeof(int));

        // create the actual device memory for the reduction storage
        amountTracker = tsneParticlesSize;
        for (int i = 0; i < reductionStorageAmount; i++)
        {
            checkCuda(cudaMalloc(&reductionStorages[i], amountTracker * sizeof(float)));
            reductionStoragesAmounts[i] = amountTracker;

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
    cudaMemcpy(sparseMatrixCSC, initSparseMatrixCSC, initSparseMatrixCSCSize * sizeof(SparseEntryCSC2D), cudaMemcpyHostToDevice);
    cudaMemcpy(sparseMatrixColumnIndexStart, initSparseMatrixColumnIndexStart, (tsneParticlesSize + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(indexTracker, indexTrackerToBuffer, tsneParticlesSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(indexTrackerPrev, indexTrackerPrevToBuffer, tsneParticlesSize * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(tsneParticles, tsneParticlesToBuffer, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);
    cudaMemcpy(tsneParticlesPrev, tsneParticlesPrevToBuffer, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);
    cudaMemcpy(tsneParticlesPrevPrev, tsneParticlesPrevPrevToBuffer, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyHostToDevice);

    // free host memory
    cudaFreeHost(indexTrackerToBuffer);
    cudaFreeHost(indexTrackerPrevToBuffer);

    cudaFreeHost(tsneParticlesToBuffer);
    cudaFreeHost(tsneParticlesPrevToBuffer);
    cudaFreeHost(tsneParticlesPrevPrevToBuffer);
}

template <class T>
NBodySolverGpuBH<T>::~NBodySolverGpuBH()
{
    cudaFree(sparseMatrixCSC);
    cudaFree(sparseMatrixColumnIndexStart);
    cudaFree(labels);

    cudaFree(indexTracker);
    cudaFree(indexTrackerPrev);

    cudaFree(tsneParticles);
    cudaFree(tsneParticlesPrev);
    cudaFree(tsneParticlesPrevPrev);

    cudaFree(nodes);
    delete[] depthSizes;
    delete[] depthStarts;

    cudaFree(xMin);
    cudaFree(yMin);
    cudaFree(xMax);
    cudaFree(yMax);
    cudaFree(repulsionSum);

    for (int i = 0; i < reductionStorageAmount; i++)
        checkCuda(cudaFree(reductionStorages[i]));
    free(reductionStorages);
    free(reductionStoragesAmounts);
}

template <class T>
void NBodySolverGpuBH<T>::timeStep()
{
    int blockSize = 256;
    int numBlocks = divUp(tsneParticlesSize, blockSize);
    
    for (int r = 0; r < 1; r++)
    {
        // obtain root min and max corners
        fillReductionX<TsneParticle2D><<<numBlocks, blockSize>>>(tsneParticles, reductionStorages[0], tsneParticlesSize);
        reductionCall(xMin, MinOp(), std::numeric_limits<float>::infinity());

        fillReductionY<TsneParticle2D><<<numBlocks, blockSize>>>(tsneParticles, reductionStorages[0], tsneParticlesSize);
        reductionCall(yMin, MinOp(), std::numeric_limits<float>::infinity());

        fillReductionX<TsneParticle2D><<<numBlocks, blockSize>>>(tsneParticles, reductionStorages[0], tsneParticlesSize);
        reductionCall(xMax, MaxOp(), -std::numeric_limits<float>::infinity());

        fillReductionY<TsneParticle2D><<<numBlocks, blockSize>>>(tsneParticles, reductionStorages[0], tsneParticlesSize);
        reductionCall(yMax, MaxOp(), -std::numeric_limits<float>::infinity());

        {
            //glm::vec2 minPos = glm::vec2(0.0f);
            //cudaMemcpy(&minPos.x, xMin, 1 * sizeof(float), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&minPos.y, yMin, 1 * sizeof(float), cudaMemcpyDeviceToHost);

            //glm::vec2 maxPos = glm::vec2(0.0f);
            //cudaMemcpy(&maxPos.x, xMax, 1 * sizeof(float), cudaMemcpyDeviceToHost);
            //cudaMemcpy(&maxPos.y, yMax, 1 * sizeof(float), cudaMemcpyDeviceToHost);

            //std::cout << "min pos is: " << glm::to_string(minPos) << std::endl;
            //std::cout << "max pos is: " << glm::to_string(maxPos) << std::endl;
        }

        // create root node
        createRoot<TsneParticle2D><<<1, 1>>>(tsneParticlesSize, xMin, yMin, xMax, yMax, nodes);

        // create tree
        for (int l = 0; l < treeDepth; l++)
        {
            //std::cout << "devide level: " << l << std::endl;
            int levelToDivide = l;
            int numberOfNodesInCurrentLevel = depthSizes[levelToDivide];
            int startIndexFirstNodeInCurrentLevel = depthStarts[levelToDivide];
            int startIndexFirstNodeInNextLevel = depthStarts[levelToDivide + 1];
            int numBlocksCreateChildren = divUp(numberOfNodesInCurrentLevel, blockSize);
            //std::cout << "launch createChildren with: " << numBlocksCreateChildren << " blocks and " << blockSize << " block size" << std::endl;

            //std::cout << "nodesSize: " << nodesSize << " ,levelToDivide: " << levelToDivide << " ,numberOfNodesInCurrentLevel: " << numberOfNodesInCurrentLevel << " ,startIndexFirstNodeInCurrentLevel: " << startIndexFirstNodeInCurrentLevel << " ,startIndexFirstNodeInNextLevel: " << startIndexFirstNodeInNextLevel << std::endl;
            createChildren<TsneParticle2D><<<numBlocksCreateChildren, blockSize>>>(tsneParticlesSize, tsneParticles, tsneParticlesPrevPrev, indexTracker, nodes, nodesSize, levelToDivide, numberOfNodesInCurrentLevel, startIndexFirstNodeInCurrentLevel, startIndexFirstNodeInNextLevel);
        }

        // test if tree is correct
        {
            //std::vector<TsneParticle2D> particlesTest(tsneParticlesSize);
            //std::vector<QuadTreeGpuBH<TsneParticle2D>> nodesTest(nodesSize);
            //cudaMemcpy(particlesTest.data(), tsneParticles, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyDeviceToHost);
            //cudaMemcpy(nodesTest.data(), nodes, nodesSize * sizeof(QuadTreeGpuBH<TsneParticle2D>), cudaMemcpyDeviceToHost);

            //int numberOfMistakes = checkTreeCorrectness(particlesTest, nodesTest, nodesTest[0], 0);
            //if (numberOfMistakes == 0)
            //{
            //    std::cout << "tree is correct!!!!!" << std::endl;
            //}
            //else
            //{
            //    std::cout << "tree is NOT correct!!!!! with: " << numberOfMistakes << " mistakes" << std::endl;
            //}
        }

        {
            int numberOfNodesInCurrentLevel = depthSizes[treeDepth];
            int startIndexFirstNodeInCurrentLevel = depthStarts[treeDepth];
            int numBlocksSumLeaves = divUp(numberOfNodesInCurrentLevel, blockSize);
            cudaSumLeavesBH<TsneParticle2D><<<numBlocksSumLeaves, blockSize>>>(tsneParticles, nodes, numberOfNodesInCurrentLevel, startIndexFirstNodeInCurrentLevel);
        }
        //std::vector<QuadTreeGpuBH<TsneParticle2D>> nodesTest(nodesSize);
        //cudaMemcpy(nodesTest.data(), nodes, nodesSize * sizeof(QuadTreeGpuBH<TsneParticle2D>), cudaMemcpyDeviceToHost);
        //float nodeSum = 0.0f;
        //for (int i = 0; i < nodesTest.size(); i++)
        //{
        //    nodeSum += nodesTest[i].totalMass;
        //}
        //std::cout << "nodeSum total: " << nodeSum << std::endl;


        for (int l = treeDepth-1; l >= 0; l--)
        {
            int numberOfNodesInCurrentLevel = depthSizes[l];
            int startIndexFirstNodeInCurrentLevel = depthStarts[l];
            int numBlocksSumNodes = divUp(numberOfNodesInCurrentLevel, blockSize);
            cudaSumNodesBH<TsneParticle2D><<<numBlocksSumNodes, blockSize>>>(nodes, nodesSize, numberOfNodesInCurrentLevel, startIndexFirstNodeInCurrentLevel);
        }
        //QuadTreeGpuBH<TsneParticle2D>* rootTest = new QuadTreeGpuBH<TsneParticle2D>();
        //cudaMemcpy(rootTest, nodes, 1 * sizeof(QuadTreeGpuBH<TsneParticle2D>), cudaMemcpyDeviceToHost);
        //std::cout << "total mass: " << rootTest->totalMass << std::endl;
        //std::cout << "centre mass: " << rootTest->centreOfMass.x << ", " << rootTest->centreOfMass.y << std::endl;
        //delete rootTest;


        // use tree to compute forces
        cudaTsneBHStepRep<TsneParticle2D><<<numBlocks, blockSize>>>(sparseMatrixCSC, sparseMatrixCSCSize, sparseMatrixColumnIndexStart, indexTracker, indexTrackerPrev, tsneParticles, tsneParticlesPrev, tsneParticlesPrevPrev, nodes, reductionStorages[0], tsneParticlesSize, learnRate, accelerationRate);

        reductionCall(repulsionSum, SumOp(), 0.0f);


        cudaTsneBHStepAtt<TsneParticle2D><<<numBlocks, blockSize>>>(sparseMatrixCSC, sparseMatrixCSCSize, sparseMatrixColumnIndexStart, indexTracker, indexTrackerPrev, tsneParticles, tsneParticlesPrev, tsneParticlesPrevPrev, repulsionSum, tsneParticlesSize, learnRate, accelerationRate);
        // push particle history down one step
        TsneParticle2D* temp = tsneParticlesPrevPrev;
        tsneParticlesPrevPrev = tsneParticlesPrev;
        tsneParticlesPrev = tsneParticles;
        tsneParticles = temp;
        cudaTsneBHStepUpd<TsneParticle2D><<<numBlocks, blockSize>>>(sparseMatrixCSC, sparseMatrixCSCSize, sparseMatrixColumnIndexStart, indexTracker, indexTrackerPrev, tsneParticles, tsneParticlesPrev, tsneParticlesPrevPrev, reductionStorages[reductionStorageAmount - 1], tsneParticlesSize, learnRate, accelerationRate);
    }
    
}

template <class T>
void NBodySolverGpuBH<T>::getTree()
{
    std::vector<QuadTreeGpuBH<TsneParticle2D>> nodesTest(nodesSize);
    cudaMemcpy(nodesTest.data(), nodes, nodesSize * sizeof(QuadTreeGpuBH<TsneParticle2D>), cudaMemcpyDeviceToHost);

    std::vector<LineSegment2D> lineSegments;
    int shownLevel = 0;
    for (int i = depthStarts[shownLevel]; i < depthStarts[shownLevel+1]; i++)
    {
        float bias = 0.0f;
        glm::vec2 LL = glm::vec2(nodesTest[i].BBcentre.x - 0.5f * nodesTest[i].BBlength, nodesTest[i].BBcentre.y - 0.5f * nodesTest[i].BBlength) + glm::vec2(bias, bias);
        glm::vec2 HL = glm::vec2(nodesTest[i].BBcentre.x + 0.5f * nodesTest[i].BBlength, nodesTest[i].BBcentre.y - 0.5f * nodesTest[i].BBlength) + glm::vec2(-bias, bias);
        glm::vec2 LH = glm::vec2(nodesTest[i].BBcentre.x - 0.5f * nodesTest[i].BBlength, nodesTest[i].BBcentre.y + 0.5f * nodesTest[i].BBlength) + glm::vec2(bias, -bias);
        glm::vec2 HH = glm::vec2(nodesTest[i].BBcentre.x + 0.5f * nodesTest[i].BBlength, nodesTest[i].BBcentre.y + 0.5f * nodesTest[i].BBlength) + glm::vec2(-bias, -bias);

        LineSegment2D lineSegment2DLLHL(LL, HL, glm::vec3(1.0f), glm::vec3(1.0f), 0);
        LineSegment2D lineSegment2DHLHH(HL, HH, glm::vec3(1.0f), glm::vec3(1.0f), 0);
        LineSegment2D lineSegment2DHHLH(HH, LH, glm::vec3(1.0f), glm::vec3(1.0f), 0);
        LineSegment2D lineSegment2DLHLL(LH, LL, glm::vec3(1.0f), glm::vec3(1.0f), 0);
        lineSegments.push_back(lineSegment2DLLHL);
        lineSegments.push_back(lineSegment2DHLHH);
        lineSegments.push_back(lineSegment2DHHLH);
        lineSegments.push_back(lineSegment2DLHLL);
    }


    std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(lineSegments);
    this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
}

template <class T>
void NBodySolverGpuBH<T>::getParticles(std::vector<TsneParticle2D>& result)
{
    cudaMemcpy(result.data(), tsneParticles, tsneParticlesSize * sizeof(TsneParticle2D), cudaMemcpyDeviceToHost);
}


template <class T>
template <typename Op>
void NBodySolverGpuBH<T>::reductionCall(float* finalResult, Op op, float identity)
{
    for (int i = 0; i < reductionStorageAmount - 1; i++)
    {
        int reductionNumBlocks = divUp(reductionStoragesAmounts[i], (reductionBlockSize * 2) * reductionGridReductionAmount);
        cudaTsneReduceBH<Op, /*replace 128 with SUMblockSize*/128> << <reductionNumBlocks, reductionBlockSize, reductionBlockSize * sizeof(float) + 1/*+ 1 is useless?*/ >> > (reductionStoragesAmounts[i], reductionStorages[i], reductionStorages[i + 1], op, identity);
    }
    int reductionNumBlocks = divUp(reductionStoragesAmounts[reductionStorageAmount - 1], (reductionBlockSize * 2) * reductionGridReductionAmount);
    cudaTsneReduceBH<Op, /*replace 128 with SUMblockSize*/128> << <reductionNumBlocks, reductionBlockSize, reductionBlockSize * sizeof(float) + 1/*+ 1 is useless?*/ >> > (reductionStoragesAmounts[reductionStorageAmount - 1], reductionStorages[reductionStorageAmount - 1], finalResult, op, identity);
}

template <class T>
int NBodySolverGpuBH<T>::checkTreeCorrectness(std::vector<TsneParticle2D>& particlesToCheck, std::vector<QuadTreeGpuBH<TsneParticle2D>>& nodesToCheck, QuadTreeGpuBH<TsneParticle2D> nodeToCheck, int currentLevel)
{
    int faults = 0;

    for (int i = nodeToCheck.firstParticleIndex; i < nodeToCheck.firstParticleIndex + nodeToCheck.particleIndexAmount; i++)
    {
        bool lessThan = glm::any(glm::lessThanEqual(particlesToCheck[i].position + glm::vec2(0.01f), nodeToCheck.BBcentre - 0.5f * nodeToCheck.BBlength));
        bool greaterThan = glm::any(glm::greaterThanEqual(particlesToCheck[i].position - glm::vec2(0.01f), nodeToCheck.BBcentre + 0.5f * nodeToCheck.BBlength));
        if (lessThan || greaterThan)
        {
            if (currentLevel == 2)
            {
                faults += 1;
            }
        }
    }

    std::cout << "---node in level: " << currentLevel << std::endl;
    if (nodeToCheck.particleIndexAmount == 0)
    {
        std::cout << "------node has zero entries" << std::endl;
    }
    else
    {
        std::cout << "------percentage of correctness: " << 100.0f * ((float)nodeToCheck.particleIndexAmount - (float)faults) / (float)nodeToCheck.particleIndexAmount << "%" << std::endl;
    }

    if (nodeToCheck.firstChildIndex != 0)
    {
        for (int i = 0; i < 4; i++)
        {
            QuadTreeGpuBH<TsneParticle2D> childToCheck = nodesToCheck[nodeToCheck.firstChildIndex + i];

            faults += checkTreeCorrectness(particlesToCheck, nodesToCheck, childToCheck, currentLevel + 1);
        }
    }

    return faults;
}


// Explicit instantiation (required for templates in .cu)
template class NBodySolverGpuBH<TsneParticle2D>;
template __global__ void cudaTsneBHStepRep<TsneParticle2D>(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, QuadTreeGpuBH<TsneParticle2D>* nodes, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate);

template __global__ void cudaTsneReduceBH<SumOp, /*replace 128 with SUMblockSize*/128>(int inAmount, const float* in, float* out, SumOp op, float identity);
template __global__ void cudaTsneReduceBH<MinOp, /*replace 128 with SUMblockSize*/128>(int inAmount, const float* in, float* out, MinOp op, float identity);
template __global__ void cudaTsneReduceBH<MaxOp, /*replace 128 with SUMblockSize*/128>(int inAmount, const float* in, float* out, MaxOp op, float identity);

template __global__ void cudaTsneBHStepAtt<TsneParticle2D>(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate);
template __global__ void cudaTsneBHStepUpd<TsneParticle2D>(SparseEntryCSC2D* sparseMatrixCSC, size_t sparseMatrixCSCSize, int* sparseMatrixColumnIndexStart, int* indexTracker, int* indexTrackerPrev, TsneParticle2D* tsneParticles, TsneParticle2D* tsneParticlesPrev, TsneParticle2D* tsneParticlesPrevPrev, float* reductionStorage, int tsneParticlesSize, float learnRate, float accelerationRate);