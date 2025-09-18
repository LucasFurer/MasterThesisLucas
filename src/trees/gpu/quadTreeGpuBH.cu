//#pragma once
//
//#include "../../trees/gpu/quadTreeGpuBH.cuh"
//
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//
//#include "../../particles/tsneParticle2D.h"
//
//
//
//template <class T>
//QuadTreeGpuBH<T>::QuadTreeGpuBH()
//{
//    BBcentre = glm::vec2(0.0f);
//    BBlength = 0.0f;
//
//    firstChildIndex = 0;
//    firstParticleIndex = 0;
//    particleIndexAmount = 0;
//
//    totalMass = 0.0f;
//}
//
//template <class T>
//QuadTreeGpuBH<T>::QuadTreeGpuBH(glm::vec2 initBBcentre, float initBBlength, unsigned int initFirstChildIndex, unsigned int initFirstParticleIndex, unsigned int initParticleIndexAmount, float initTotalMass)
//{
//    BBcentre = initBBcentre;
//    BBlength = initBBlength;
//
//    firstChildIndex = initFirstChildIndex;
//    firstParticleIndex = initFirstParticleIndex;
//    particleIndexAmount = initParticleIndexAmount;
//
//    totalMass = initTotalMass;
//}
//
//template <class T>
//QuadTreeGpuBH<T>::~QuadTreeGpuBH()
//{
//
//}
//
//template <class T>
//void QuadTreeGpuBH<T>::createTree(QuadTreeGpuBH<T> node)
//{
//
//}
//
//
//
//template class QuadTreeGpuBH<TsneParticle2D>;