#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // remove
#include <glm/gtc/type_ptr.hpp> // remove

#include <cuda_runtime.h>


template <class T>
class QuadTreeGpuBH
{
public:
    glm::vec2 BBcentre;
    float BBlength;

    unsigned int firstChildIndex; // firstChildIndex + 3 is index of last child
    unsigned int firstParticleIndex; // firstParticleIndex + particleIndexAmount is index of last particle
    unsigned int particleIndexAmount; // amount of particles in node

    float totalMass;
    glm::vec2 centreOfMass;

    __host__ __device__
    QuadTreeGpuBH()
    {

    }

    __host__ __device__
    QuadTreeGpuBH(glm::vec2 initBBcentre, float initBBlength, unsigned int initFirstChildIndex, unsigned int initFirstParticleIndex, unsigned int initParticleIndexAmount, float initTotalMass, glm::vec2 initCentreOfMass)
    {
        BBcentre = initBBcentre;
        BBlength = initBBlength;

        firstChildIndex = initFirstChildIndex;
        firstParticleIndex = initFirstParticleIndex;
        particleIndexAmount = initParticleIndexAmount;

        totalMass = initTotalMass;
        centreOfMass = initCentreOfMass;
    }

    __host__ __device__
    ~QuadTreeGpuBH()
    {

    }
};


//template <class T>
//class QuadTreeGpuBH
//{
//public:
//    glm::vec2 BBcentre;
//    float BBlength;
//
//    unsigned int firstChildIndex; // firstChildIndex + 3 is index of last child
//    unsigned int firstParticleIndex; // firstParticleIndex + particleIndexAmount is index of last particle
//    unsigned int particleIndexAmount; // amount of particles in node
//
//    float totalMass;
//
//
//    QuadTreeGpuBH();
//    QuadTreeGpuBH(glm::vec2 initBBcentre, float initBBlength, unsigned int initFirstChildIndex, unsigned int initFirstParticleIndex, unsigned int initParticleIndexAmount, float initTotalMass);
//    ~QuadTreeGpuBH();
//
//    void createTree(QuadTreeGpuBH<T> node);
//};