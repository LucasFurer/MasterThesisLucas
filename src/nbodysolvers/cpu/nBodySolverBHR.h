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

    NBodySolverBHR(std::function<void(double&, QuadTreeBarnesHutReverse<T>*, T&)> initKernelNP, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, float initTheta)
    {
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    {
        total = 0.0f;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBHR(total, &root, points[i], this->theta);
        }

        cascadeValues(points, &root, glm::vec2(0.0f));
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
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
    void traverseBHR(double& total, QuadTreeBarnesHutReverse<T>* node, T point, float theta)
    {
        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 diff = point.position - node->centreOfMass;

        
        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelNP(total, node, point);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, point.position)))
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


    void cascadeValues(std::vector<T>& points, QuadTreeBarnesHutReverse<T>* node, glm::vec2 accumulatedVal)
    {
        glm::vec2 newAccumulatedVal = accumulatedVal + node->acceleration;

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
    glm::vec2 diff = sinkNode->centreOfMass - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = (1.0f / (1.0f + (dist * dist)));
    total += sinkNode->totalMass * forceDecay;

    sinkNode->acceleration += forceDecay * forceDecay * diff;
}

void TSNEBHRPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += 1.0f * forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}


//glm::vec2 GRAVITYbarnesHutReverseParticleNodeKernal(float* accumulator, Particle2D i, QuadTreeBarnesHutReverse<Particle2D>* j)
//{
//    float softening = 0.1f;
//
//    glm::vec2 nodeDiff = j->centreOfMass - i.position;
//    float distance = glm::length(nodeDiff);
//
//    float oneOverDistance = (1.0f / (softening + distance));
//    
//    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
//}
//
//glm::vec2 GRAVITYbarnesHutReverseParticleParticleKernal(float* accumulator, Particle2D i, Particle2D j)
//{
//    float softening = 0.1f;
//
//    glm::vec2 diff = j.position - i.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = 1.0f / (softening + distance);
//
//    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}