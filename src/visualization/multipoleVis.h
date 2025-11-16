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
	glm::vec2 clusterOfsett = glm::vec2(10.0f, 0.0f);
	std::vector<EmbeddedPoint> smallCluster
	{
		EmbeddedPoint(glm::vec2(-1.0f,  0.5f), 0),
		EmbeddedPoint(glm::vec2(-1.0f, -0.5f), 0),
		EmbeddedPoint(glm::vec2( 1.0f,  0.0f), 0)
	};
	std::vector<EmbeddedPoint> smallClusterAngled(smallCluster.size());

	std::vector<EmbeddedPoint> centeredCluster
	{
		EmbeddedPoint(glm::vec2(-1.0f,  0.5f), 0),
		EmbeddedPoint(glm::vec2(-1.0f, -0.5f), 0),
		EmbeddedPoint(glm::vec2(1.0f,  0.0f), 0)
	};

	void initMultipoleVisData()
	{
		for (int i = 0; i < smallCluster.size(); i++)
		{
			smallClusterAngled[i] = visHelp::rotate(smallCluster[i], -45.0f);
			smallClusterAngled[i].position -= clusterOfsett;

			smallCluster[i].position += clusterOfsett;
		}
	}

	void testFMMtoBH()
	{
		std::vector<glm::vec2> clusterAaccNaive(3, glm::vec2(0.0f));
		std::vector<glm::vec2> clusterBaccNaive(3, glm::vec2(0.0f));
		visHelp::NodeNodeNaive(smallCluster, clusterAaccNaive, smallClusterAngled, clusterBaccNaive);

		std::vector<glm::vec2> clusterAaccBH(3, glm::vec2(0.0f));
		std::vector<glm::vec2> clusterBaccBH(3, glm::vec2(0.0f));
		visHelp::NodeNodeBH(smallCluster, clusterAaccBH, smallClusterAngled, clusterBaccBH);

		std::vector<glm::vec2> clusterAaccFMM(3, glm::vec2(0.0f));
		std::vector<glm::vec2> clusterBaccFMM(3, glm::vec2(0.0f));
		visHelp::NodeNodeFMM(smallCluster, clusterAaccFMM, smallClusterAngled, clusterBaccFMM);

		std::cout << "MSE of BH: " << visHelp::getMSE(clusterAaccBH, clusterAaccNaive) << std::endl;
		std::cout << "MSE of FMM: " << visHelp::getMSE(clusterAaccFMM, clusterAaccNaive) << std::endl;
	}

	void makeFieldAndErrorPNG()
	{

	}
};