#pragma once

template <typename T>
class NBodySolver
{
public:
	std::vector<LineSegment2D> lineSegments;
	Buffer* boxBuffer = new Buffer();
	int showLevel = 0;

	int maxChildren;
	float theta;

	virtual void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* points) = 0;

	~NBodySolver()
	{
		delete boxBuffer; // eurhmmmm will opengl complain on linux when i delete the dynamic buffers since the glfw context might be destroyed before the destructor is called leading to the opengl attributes not being able to be cleaned up properly
	}
};