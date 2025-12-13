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

template <typename T>
class NBodySolverBH : public NBodySolver<T>
{
public:
    QuadTree<T> root;

    std::function<void(double&, T&, QuadTree<T>*)> kernelPN;
    std::function<void(double&, T&, T&)> kernelPP;

    NBodySolverBH() {}

    NBodySolverBH(std::function<void(double&, T&, QuadTree<T>*)> initKernelPN, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, float initTheta)
    {
        kernelPN = initKernelPN;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    {
        total = 0.0;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBH(total, points[i], &root, this->theta);
        }
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
    {
        root = std::move(QuadTree<T>(this->maxChildren, &points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }

private:
    void traverseBH(double& total, T& point, QuadTree<T>* node, float theta)
    {
        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 diff = point.position - node->centreOfMass;


        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(diff) < theta)// && (glm::any(glm::lessThan(point.position, node->lowestCorner)) || glm::any(glm::greaterThan(point.position, node->highestCorner))))
        {

            kernelPN(total, point, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if ((*node->allParticles)[i].ID != point.ID) // self intersection test
                {

                    kernelPP(total, point, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTree<T>* child : node->children)
            {

                traverseBH(total, point, child, theta);

            }
        }
    }

};


void TSNEBHPNKernel(double& total, TsnePoint2D& sinkPoint, QuadTree<TsnePoint2D>* sourceNode)
{
    glm::vec2 diff = sinkPoint.position - sourceNode->centreOfMass;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += sourceNode->totalMass * forceDecay;

    sinkPoint.derivative += sourceNode->totalMass * forceDecay * forceDecay * diff;
}

void TSNEBHPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}