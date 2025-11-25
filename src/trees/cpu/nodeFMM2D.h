#pragma once

#include <glm/glm.hpp>
#include <Fastor/Fastor.h>

class NodeFMM2D
{
public:
	glm::vec2 BBcentre;
	float BBlength;

	unsigned int firstChildIndex; // firstChildIndex + 3 is index of last child
	unsigned int firstParticleIndex; // firstParticleIndex + particleIndexAmount is index of last particle
	unsigned int particleIndexAmount; // amount of particles in node

	glm::vec2 centreOfMass;

	float M0;
	Fastor::Tensor<float, 2, 2> M2;

	Fastor::Tensor<float, 2> C1;
	Fastor::Tensor<float, 2, 2> C2;
	Fastor::Tensor<float, 2, 2, 2> C3;

	NodeFMM2D() :
		BBcentre(0.0f),
		BBlength(0.0f),
		firstChildIndex(0),
		firstParticleIndex(0),
		particleIndexAmount(0),
		centreOfMass(0.0f),
		M0(0.0f),
		M2(0.0f),
		C1(0.0f),
		C2(0.0f),
		C3(0.0f)
	{}
};
