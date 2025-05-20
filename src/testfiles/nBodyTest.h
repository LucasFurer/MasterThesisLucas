#pragma once

#include "../trees/quadtree.h"
#include <functional>
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
class NBodySolverNaiveTest
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer;// = new Buffer();
    int showLevel = 0;

    int maxChildren;
    float theta;

    std::function<void(float*, std::vector<T>*, int, int, std::vector<glm::vec2>*)> kernel;

    NBodySolverNaiveTest()
    {

    }

    NBodySolverNaiveTest(std::function<void(float*, std::vector<T>*, int, int, std::vector<glm::vec2>*)> initKernel)
    {
        kernel = initKernel;
    }

private:

};