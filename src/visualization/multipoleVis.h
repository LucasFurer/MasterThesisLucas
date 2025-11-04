#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

#include "../particles/embeddedPoint.h"
#include "../trees/cpu/quadtreeNodeFMMiter.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
#include "visHelp.h"

namespace MultipoleVis
{
	float clusterXofsett = 3.0f;
	std::vector<EmbeddedPoint> smallCluster
	{
		EmbeddedPoint(glm::vec2(-1.0f + clusterXofsett,  0.5f), 0),
		EmbeddedPoint(glm::vec2(-1.0f + clusterXofsett, -0.5f), 0),
		EmbeddedPoint(glm::vec2( 1.0f + clusterXofsett,  0.0f), 0)
	};
	std::vector<EmbeddedPoint> smallClusterAngled
	{
		visHelp::rotate(smallCluster[0], -45.0f),
		visHelp::rotate(smallCluster[1], -45.0f),
		visHelp::rotate(smallCluster[2], -45.0f)
	};

	void NodeNodeFMM(std::vector<EmbeddedPoint>& clusterA, std::vector<EmbeddedPoint>& clusterB)
	{
		QuadTreeNodeFMMiter<EmbeddedPoint> nodeA(3, clusterA);
		QuadTreeNodeFMMiter<EmbeddedPoint> nodeB(3, clusterB);

		float dummy = 0.0f;
		TSNEFMMiterInteractionKernalNodeNode(&dummy, &nodeA, &nodeB);
		nodeA.divideC();
		nodeB.divideC();
		//nodeA.applyForces();
		//nodeA.applyForces();


		glm::vec2 accA(0.0f);
		glm::vec2 accB(0.0f);
	}


};