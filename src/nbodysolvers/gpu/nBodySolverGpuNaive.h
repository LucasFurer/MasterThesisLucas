/*
#pragma once

#include "../../nbodysolvers/gpu/nBodySolverGpu.h"
#include "../../nbodysolvers/gpu/nBodySolverGpuNaive.cuh"

//template <typename T>
//extern void cudaComputeForcesNaive(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker);
//template <typename T>
//extern void cudaComputeForcesNaive(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker);

template <typename T>
class NBodySolverGpuNaive : public NBodySolverGpu<T>
{
public:
	std::function<void(float&, T&, T&)> kernel;

	NBodySolverGpuNaive() {}

	NBodySolverGpuNaive(std::function<void(float&, T&, T&)> initKernel)
	{
		kernel = initKernel;
	}

	void solveNbody(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker) override
	{
		updateTree(particles, indexTracker);

		accumulator = 0.0f;
		for (T& particle : particles)
			particle.derivative = glm::vec2(0.0f);


		//for (int i = 0; i < particles.size(); i++)
		//{
		//	for (int j = 0; j < particles.size(); j++)
		//	{
		//		if (i != j)
		//		{

		//			kernel(accumulator, particles[indexTracker[i]], particles[indexTracker[j]]);

		//		}
		//	}
		//}

		cudaComputeForcesNaive<T>(accumulator, particles, indexTracker);
	}

	void updateTree(std::vector<T>& particles, std::vector<int>& indexTracker) override
	{
		//for (int i = 0; i < particles.size()-1; i++)
		//{
		//	swapParticle(particles, indexTracker, i, i + 1);
		//}
	}

	inline void swapParticle(std::vector<T>& particles, std::vector<int>& indexTracker, int i, int j)
	{
		indexTracker[particles[i].ID] = j;
		indexTracker[particles[j].ID] = i;

		std::swap(particles[i], particles[j]);
	}
};



void TSNEGPUnaiveKernal(float& accumulator, TsneParticle2D& passiveParticle, TsneParticle2D& activeParticle)
{
	float softening = 1.0f;

	glm::vec2 diff = activeParticle.position - passiveParticle.position;
	float distance = glm::length(diff);

	float oneOverDistance = 1.0f / (softening + (distance * distance));
	accumulator += 1.0f * oneOverDistance;

	passiveParticle.derivative += oneOverDistance * oneOverDistance * diff;
}
*/