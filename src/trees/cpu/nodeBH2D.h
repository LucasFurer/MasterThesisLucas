#pragma once

#include <glm/glm.hpp>

class NodeBH2D
{
	glm::vec2 BBcentre;
	float BBlength;

	unsigned int firstChildIndex; // firstChildIndex + 3 is index of last child
	unsigned int firstParticleIndex; // firstParticleIndex + particleIndexAmount is index of last particle
	unsigned int particleIndexAmount; // amount of particles in node

	float totalMass;
	glm::vec2 centreOfMass;

	NodeBH2D() 
	{
		BBcentre = glm::vec2(0.0f);
		BBlength = 0.0f;

		firstChildIndex = 0;
		firstParticleIndex = 0;
		particleIndexAmount = 0;

		totalMass = 0.0f;
		centreOfMass = glm::vec2(0.0f);
	}
};
