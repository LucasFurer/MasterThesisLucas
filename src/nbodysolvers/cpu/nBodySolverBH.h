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

    NBodySolverBH(std::function<void(double&, T&, QuadTree<T>*)> initKernelPN, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, double initTheta)
    {
        kernelPN = initKernelPN;
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
            traverseBH(total, points[i], &root, this->theta);
        }
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
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
    void traverseBH(double& total, T& point, QuadTree<T>* node, double theta)
    {
        double l = node->highestCorner.x - node->lowestCorner.x;
        glm::dvec2 diff = point.position - node->centreOfMass;


        if (l / glm::length(diff) < theta)// && (glm::any(glm::lessThan(point.position, node->lowestCorner)) || glm::any(glm::greaterThan(point.position, node->highestCorner))))
        {

            kernelPN(total, point, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (&(*node->allParticles)[i] != &point) // self intersection test
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
    //std::cout << "im doing a PN interaction" << std::endl;

    glm::dvec2 diff = sinkPoint.position - sourceNode->centreOfMass;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += sourceNode->totalMass * forceDecay;

    sinkPoint.derivative += sourceNode->totalMass * forceDecay * forceDecay * diff;
}

void TSNEBHPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}