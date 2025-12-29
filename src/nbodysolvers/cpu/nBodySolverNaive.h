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
    std::function<void(double&, T&, T&)> kernel;

    NBodySolverNaive() {}

    NBodySolverNaive(std::function<void(double&, T&, T&)> initKernel)
    {
        kernel = initKernel;
    }

    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    {
        total = 0.0;

        if (kernel)
        {
            for (int i = 0; i < points.size(); i++)
            {
                for (int j = 1; j < points.size(); j++)
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

void TSNEnaiveKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += static_cast<double>(2.0f * forceDecay);

    sinkPoint.derivative += forceDecay * forceDecay * diff;
    sourcePoint.derivative += forceDecay * forceDecay * -diff;
}