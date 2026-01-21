#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <Fastor/Fastor.h>
#include <string>

class NodeFMM_MORTON_2D
{
public:
	glm::dvec2 BBcentre;
	double BBlength;

	unsigned int firstChildIndex; // firstChildIndex + 3 is index of last child
	unsigned int firstParticleIndex; // firstParticleIndex + particleIndexAmount is index of last particle
	unsigned int particleIndexAmount; // amount of particles in node

	glm::dvec2 centreOfMass;

	double M0;
	Fastor::Tensor<double, 2, 2> M2;

	Fastor::Tensor<double, 2> C1;
	Fastor::Tensor<double, 2, 2> C2;
	Fastor::Tensor<double, 2, 2, 2> C3;

	NodeFMM_MORTON_2D() :
		BBcentre(0.0),
		BBlength(0.0),
		firstChildIndex(0u),
		firstParticleIndex(0u),
		particleIndexAmount(0u),
		centreOfMass(0.0),
		M0(0.0),
		M2(0.0),
		C1(0.0),
		C2(0.0),
		C3(0.0)
	{}

	//std::string toString()
	//{
	//	std::string result =
	//		"    BBcentre: " + glm::to_string(BBcentre) + "\n" +
	//		"    BBlength: " + std::to_string(BBlength) + "\n" +
	//		"    firstChildIndex: " + std::to_string(firstChildIndex) + "\n" +
	//		"    firstParticleIndex: " + std::to_string(firstParticleIndex) + "\n" +
	//		"    particleIndexAmount: " + std::to_string(particleIndexAmount) + "\n" +
	//		"    centreOfMass: " + glm::to_string(centreOfMass)
	//	;

	//	return result;
	//}
};
