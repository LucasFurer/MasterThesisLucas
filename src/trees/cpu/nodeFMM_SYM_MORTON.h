#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <Fastor/Fastor.h>
#include <string>
#include <cstddef>

class Node_FMM_SYM_MORTON_2D
{
public:
	glm::dvec2 BBcentre{0.0};
	double BBlength{0.0};

	std::vector<Node_FMM_SYM_MORTON_2D*> nodeChildren;
	std::size_t firstParticleIndex{0}; // firstParticleIndex + particleIndexAmount is index of last particle
	std::size_t particleIndexAmount{0}; // amount of particles in node

	glm::dvec2 centreOfMass{0.0};

	double M0{0.0};
	Fastor::Tensor<double, 2, 2> M2{0.0};

	Fastor::Tensor<double, 2> C1{0.0};
	Fastor::Tensor<double, 2, 2> C2{0.0};
	Fastor::Tensor<double, 2, 2, 2> C3{0.0};

	Node_FMM_SYM_MORTON_2D() {}
};