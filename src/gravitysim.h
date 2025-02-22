#pragma once
#include <vector>
#include "particles/particle3D.h"
#include "buffer.h"

class GravitySim
{
public:
	std::vector<Particle2D> particles;
	std::vector<glm::vec2> accelerations;
	Buffer* particlesBuffer;

	GravitySim(int particleAmount)
	{

	}

private:
};