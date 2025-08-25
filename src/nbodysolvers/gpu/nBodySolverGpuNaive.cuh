#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <GLFW/glfw3.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"



template <class T>
class NBodySolverGpuNaive
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


    NBodySolverGpuNaive(int initTsneParticlesSize, SparseEntryCSC2D* initSparseMatrixCSC, size_t initSparseMatrixCSCSize, int* initSparseMatrixColumnIndexStart, int* initLabels, float initLearnRate, float initAccelerationRate);
    ~NBodySolverGpuNaive();

    void timeStep();
    void getParticles(std::vector<TsneParticle2D>& result);
};