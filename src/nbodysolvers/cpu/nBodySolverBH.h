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

    std::function<void(float&, T&, QuadTree<T>*)> kernelPN;
    std::function<void(float&, T&, T&)> kernelPP;

    NBodySolverBH() {}

    NBodySolverBH(std::function<void(float&, T&, QuadTree<T>*)> initKernelPN, std::function<void(float&, T&, T&)> initKernelPP, int initMaxChildren, float initTheta)
    {
        kernelPN = initKernelPN;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(float& total, std::vector<T>& points) override
    {
        total = 0.0f;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBH(total, points[i], &root, this->theta);
        }
    }

    void updateTree(std::vector<T>& points) override
    {
        root = std::move(QuadTree<T>(this->maxChildren, &points));
    }

    std::vector<LineSegment2D> getNodesBufferData(int level) override
    {
        return std::vector<LineSegment2D>();

        //this->lineSegments.clear();
        //root.getLineSegments(this->lineSegments, 0, this->showLevel);
        //std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        //this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    void traverseBH(float& total, T& point, QuadTree<T>* node, float theta)
    {
        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = point.position - node->centreOfMass;


        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelPN(total, point, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, point.position)))
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


void TSNEBHPNKernel(float& total, TsnePoint2D& sinkPoint, QuadTree<TsnePoint2D>* sourceNode)
{
    glm::vec2 diff = sinkPoint.position - sourceNode->centreOfMass;
    float dist = glm::length(diff);

    float forceDecay = (1.0f / (1.0f + (dist * dist)));
    total += sourceNode->totalMass * forceDecay;

    sinkPoint.derivative += sourceNode->totalMass * forceDecay * forceDecay * diff;
}

void TSNEBHPPKernel(float& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += 1.0f * forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}



//glm::vec2 GRAVITYbarnesHutParticleNodeKernel(float* accumulator, Particle2D i, QuadTree<Particle2D>* j)
//{
//    float softening = 0.1f;
//
//    glm::vec2 nodeDiff = i.position - j->centreOfMass;
//    float parCentreDistance = glm::length(nodeDiff);
//
//    float oneOverDistance = (1.0f / (softening + parCentreDistance));
//
//    return -j->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
//}
//
//glm::vec2 GRAVITYbarnesHutParticleParticleKernel(float* accumulator, Particle2D i, Particle2D j)
//{
//    float softening = 0.1f;
//
//    glm::vec2 diff = i.position - j.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = 1.0f / (softening + distance);
//
//    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}