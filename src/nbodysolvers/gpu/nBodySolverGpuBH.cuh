#pragma once

#include <vector>
#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"
#include "../../nbodysolvers/gpu/nBodySolverGpu.cuh"
#include "../../trees/gpu/quadTreeGpuBH.cuh"



template <class T>
class NBodySolverGpuBH : public NBodySolverGpu<T>
{
public:
    SparseEntryCSC2D* sparseMatrixCSC;
    size_t sparseMatrixCSCSize;
    int* sparseMatrixColumnIndexStart;

    int* labels;

    int* indexTracker;
    int* indexTrackerPrev;

    TsneParticle2D* tsneParticles;
    TsneParticle2D* tsneParticlesPrev;
    TsneParticle2D* tsneParticlesPrevPrev;

    const int reductionBlockSize = 128;
    const int reductionGridReductionAmount = 64;
    int reductionStorageAmount;
    float** reductionStorages;
    int* reductionStoragesAmounts;

    int nodesSize;
    int treeDepth;
    int* depthSizes;
    int* depthStarts;

    QuadTreeGpuBH<TsneParticle2D>* nodes;

    float* xMin;
    float* yMin;
    float* xMax;
    float* yMax;
    float* repulsionSum;

    int tsneParticlesSize;
    float learnRate;
    float accelerationRate;


    NBodySolverGpuBH(int initTsneParticlesSize, SparseEntryCSC2D* initSparseMatrixCSC, size_t initSparseMatrixCSCSize, int* initSparseMatrixColumnIndexStart, std::vector<uint8_t>& initLabels, float initLearnRate, float initAccelerationRate, int treeDepth);
    ~NBodySolverGpuBH() override;


    void timeStep() override;
    void getParticles(std::vector<TsneParticle2D>& result) override;
    void getTree() override;

    template <typename Op>
    void reductionCall(float* finalResult, Op op, float identity);

    int checkTreeCorrectness(std::vector<TsneParticle2D>& particlesToCheck, std::vector<QuadTreeGpuBH<TsneParticle2D>>& nodesToCheck, QuadTreeGpuBH<TsneParticle2D> nodeToCheck, int currentLevel);
};