#pragma once

#include <vector>
#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"
#include "../../nbodysolvers/gpu/nBodySolverGpu.cuh"



template <class T>
class NBodySolverGpuNaive : public NBodySolverGpu<T>
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

    const int SUMblockSize = 128;
    const int SUMgridReductionAmount = 64;
    int sumStorageAmount;
    float** sumStorages;
    int* sumStoragesAmounts;

    int tsneParticlesSize;
    float learnRate;
    float accelerationRate;


    NBodySolverGpuNaive(int initTsneParticlesSize, SparseEntryCSC2D* initSparseMatrixCSC, size_t initSparseMatrixCSCSize, int* initSparseMatrixColumnIndexStart, std::vector<uint8_t>& initLabels, float initLearnRate, float initAccelerationRate);
    ~NBodySolverGpuNaive() override;

    void timeStep() override;
    void getParticles(std::vector<TsneParticle2D>& result) override;
    void getTree() override;
};