#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "../../openGLhelper/buffer.h"


template <typename T>
class NBodySolver
{
public:
	int maxChildren;
	double theta;
	double cell_size = 0.0;

	#ifdef INDEX_TRACKER
	virtual void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) = 0;
	#else
	virtual void solveNbody(double& total, std::vector<T>& points) = 0;
	#endif
	virtual void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) = 0;
	virtual std::vector<VertexPos2Col3> getNodesBufferData(int level) = 0;


	virtual ~NBodySolver() = default;
};