#pragma once

template <typename T>
class NBodySolverGpu
{
public:
	//std::vector<LineSegment2D> lineSegments;
	//Buffer* boxBuffer = new Buffer();
	//int showLevel = 0;

	int maxChildren;
	float theta;

	virtual void solveNbody(float& accumulator, std::vector<T>& particles, std::vector<int>& indexTracker) = 0;
	virtual void updateTree(std::vector<T>& particles, std::vector<int>& indexTracker) = 0;


	virtual ~NBodySolverGpu()
	{
		//delete boxBuffer;
	}

	void cleanup()
	{
		//boxBuffer->cleanup();
	}
};