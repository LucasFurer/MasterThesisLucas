#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "../../openGLhelper/buffer.h"


template <typename T>
class NBodySolver
{
public:
	//std::vector<T> points;

	//std::vector<LineSegment2D> lineSegments;
	//Buffer* boxBuffer = new Buffer();
	//int showLevel = 0;

	int maxChildren;
	float theta;

	virtual void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) = 0;
	virtual void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) = 0;
	virtual std::vector<VertexPos2Col3> getNodesBufferData(int level) = 0;


	virtual ~NBodySolver() = default;

	//void cleanup()
	//{
	//	//boxBuffer->cleanup();
	//}
};