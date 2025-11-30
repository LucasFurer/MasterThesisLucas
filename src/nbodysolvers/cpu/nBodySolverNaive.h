#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtree.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/Particle2D.h"
#include "../../particles/tsnePoint2D.h"

template <typename T>
class NBodySolverNaive : public NBodySolver<T>
{
public:
    std::function<void(float&, T&, T&)> kernel;

    NBodySolverNaive() {}

    NBodySolverNaive(std::function<void(float&, T&, T&)> initKernel)
    {
        kernel = initKernel;
    }

    void solveNbody(float& total, std::vector<T>& points) override
    {
        total = 0.0f;

        if (kernel)
        {
            for (int i = 0; i < points.size(); i++)
            {
                for (int j = 0; j < points.size(); j++)
                {
                    if (i != j)
                    {

                        kernel(total, points[i], points[j]);

                    }
                }
            }
        }
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override {}

    std::vector<VertexPos2Col3> getNodesBufferData(int level) override { return std::vector<VertexPos2Col3>(); }

private:

};

void TSNEnaiveKernel(float& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}

//void GRAVITYnaiveKernel(float* accumulator, std::vector<Particle2D>* embeddedPoints, int i, int j, std::vector<glm::vec2>* forces)
//{
//    float softening = 0.1f; // should be 1.0f for t-SNE
//
//    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = 1.0f / (softening + distance);
//
//    (*forces)[i] += oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
