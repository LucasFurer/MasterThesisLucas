#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

#include "../particles/embeddedPoint.h"
#include "../trees/cpu/quadtreeNodeFMMiter.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"

namespace visHelp
{
	EmbeddedPoint rotate(EmbeddedPoint P, float degrees)
	{
		float radiance = 2.0f * std::numbers::pi_v<float> *degrees / 360.0f;
		return EmbeddedPoint
		(
			glm::vec2
			(
				P.position.x * std::cos(radiance) - P.position.y * std::sin(radiance),
				P.position.x * std::cos(radiance) - P.position.y * std::sin(radiance)
			),
			P.label
		);
	}
}