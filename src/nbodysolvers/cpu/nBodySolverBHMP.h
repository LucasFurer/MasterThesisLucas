#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>


#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtreemultipole.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"

template <typename T>
class NBodySolverBHMP : public NBodySolver<T>
{
public:
    QuadTreeMultiPole<T> root;

    std::function<void(double&, T&, QuadTreeMultiPole<T>*)> kernelPN;
    std::function<void(double&, T&, T&)> kernelPP;

    NBodySolverBHMP() {}

    NBodySolverBHMP(std::function<void(double&, T&, QuadTreeMultiPole<T>*)> initKernelPN, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, double initTheta)
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
            traverseBHMP(total, points[i], &root, this->theta);
        }
    }

    void updateTree(std::vector<T>& embeddedPoints, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        root = std::move(QuadTreeMultiPole<T>(this->maxChildren, &embeddedPoints));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }

private:
    void traverseBHMP(double& total, T& point, QuadTreeMultiPole<T>* node, double theta)
    {
        double l = node->highestCorner.x - node->lowestCorner.x;
        glm::dvec2 diff = point.position - node->centreOfMass;

        if (l / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelPN(total, point, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (&(*node->allParticles)[i] != &point)
                {

                    kernelPP(total, point, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTreeMultiPole<T>* child : node->children)
            {

                traverseBHMP(total, point, child, theta);

            }
        }

    }

};



void TSNEBHMPPNKernel(double& total, TsnePoint2D& sinkPoint, QuadTreeMultiPole<TsnePoint2D>* sourceNode)
{
    glm::dvec2 R = sinkPoint.position - sourceNode->centreOfMass;
    double sq_r = R.x * R.x + R.y * R.y;
    double rS = 1.0 + sq_r;

    double D1 = 1.0 / (rS * rS);
    double D2 = -4.0 / (rS * rS * rS);
    double D3 = 24.0 / (rS * rS * rS * rS);
    total += sourceNode->totalMass / rS;

    double MB0 = sourceNode->totalMass;
    Fastor::Tensor<double, 2, 2> MB2 = sourceNode->quadrupole;
    Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;

    double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<double, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    sinkPoint.derivative += glm::dvec2(C1(0), C1(1));
}

void TSNEBHMPPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}