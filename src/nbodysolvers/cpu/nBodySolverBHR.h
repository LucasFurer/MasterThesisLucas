#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadTreeBarnesHutReverse.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"


template <typename T>
class NBodySolverBHR : public NBodySolver<T>
{
public:
    QuadTreeBarnesHutReverse<T> root;

    std::function<void(double&, QuadTreeBarnesHutReverse<T>*, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    NBodySolverBHR() {}

    NBodySolverBHR(std::function<void(double&, QuadTreeBarnesHutReverse<T>*, T&)> initKernelNP, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, double initTheta)
    {
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    #ifdef INDEX_TRACKER
    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
	#else
    void solveNbody(double& total, std::vector<T>& points) override
	#endif
    {
        total = 0.0;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBHR(total, &root, points[i], this->theta);
        }

        cascadeValues(points, &root, glm::dvec2(0.0));
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        root = std::move(QuadTreeBarnesHutReverse<T>(this->maxChildren, &points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }

private:
    void traverseBHR(double& total, QuadTreeBarnesHutReverse<T>* node, T point, double theta)
    {
        double l = node->highestCorner.x - node->lowestCorner.x;
        glm::dvec2 diff = point.position - node->centreOfMass;

        
        if (l / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelNP(total, node, point);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (&(*node->allParticles)[i] != &point)
                {

                    kernelPP(total, (*node->allParticles)[i], point);

                }
            }
        }
        else
        {
            for (QuadTreeBarnesHutReverse<T>* child : node->children)
            {

                traverseBHR(total, child, point, theta);

            }
        }
        
    }


    void cascadeValues(std::vector<T>& points, QuadTreeBarnesHutReverse<T>* node, glm::dvec2 accumulatedVal)
    {
        glm::dvec2 newAccumulatedVal = accumulatedVal + node->acceleration;

        if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {

                points[i].derivative += newAccumulatedVal;

            }
        }
        else
        {
            for (QuadTreeBarnesHutReverse<T>* childQuadTree : node->children)
            {

                cascadeValues(points, childQuadTree, newAccumulatedVal);

            }
        }
    }

};


void TSNEBHRNPKernel(double& total, QuadTreeBarnesHutReverse<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkNode->centreOfMass - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += sinkNode->totalMass * forceDecay;

    sinkNode->acceleration += forceDecay * forceDecay * diff;
}

void TSNEBHRPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}