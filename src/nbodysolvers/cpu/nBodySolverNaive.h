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
    std::function<void(float&, std::vector<T>&, int, int)> kernel;

    NBodySolverNaive() {}

    NBodySolverNaive(std::function<void(float&, std::vector<T>&, int, int)> initKernel)
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

                        kernel(total, points, i, j);

                    }
                }
            }
        }
    }

    void updateTree(std::vector<T>& points) override {}

    std::vector<LineSegment2D> getNodesBufferData(int level) override { return std::vector<LineSegment2D>(); }

private:

};

void TSNEnaiveKernal(float& total, std::vector<TsnePoint2D>& points, int i, int j)
{
    glm::vec2 diff = points[i].position - points[j].position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += forceDecay;

    points[i].derivative += forceDecay * forceDecay * diff;
}

//void GRAVITYnaiveKernal(float* accumulator, std::vector<Particle2D>* embeddedPoints, int i, int j, std::vector<glm::vec2>* forces)
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
