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

struct GpuTsneParticle2D
{
    float2 position;
    float2 derivative;
    int label;
    int ID;

    GpuTsneParticle2D()
    {
        position = make_float2(0.0f, 0.0f);
        derivative = make_float2(0.0f, 0.0f);
        label = 0;
        ID = 0;
    }

    GpuTsneParticle2D(glm::vec2 initPosition, glm::vec2 initDerivative, int initLabel, int initID)
    {
        position = make_float2(initPosition.x, initPosition.y);
        derivative = make_float2(initDerivative.x, initDerivative.y);
        label = initLabel;
        ID = initID;
    }
};

//template <typename T> 
__global__
void cudaComputeForcesKernel(float* accumulator, GpuTsneParticle2D* particles, int* indexTracker);

template <typename T>
void cudaComputeForcesNaive(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker);