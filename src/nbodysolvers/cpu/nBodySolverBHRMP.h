#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadTreeBarnesHutReverseMultiPole.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"

template <typename T>
class NBodySolverBHRMP : public NBodySolver<T>
{
public:
    QuadTreeBarnesHutReverseMultiPole<T> root;

    std::function<void(double&, QuadTreeBarnesHutReverseMultiPole<T>*, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    NBodySolverBHRMP() {}

    NBodySolverBHRMP(std::function<void(double&, QuadTreeBarnesHutReverseMultiPole<T>*, T&)> initKernelNP, std::function<void(double&, T&, T&)> initKernelPP, int initMaxChildren, double initTheta)
    {
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        total = 0.0;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBHRMP(total, &root, points[i], this->theta);
        }

        root.applyForces(points);
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        root = std::move(QuadTreeBarnesHutReverseMultiPole<T>(this->maxChildren, &points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }
    
private:
    void traverseBHRMP(double& total, QuadTreeBarnesHutReverseMultiPole<T>* node, T& point, double theta)
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
            for (QuadTreeBarnesHutReverseMultiPole<T>* child : node->children)
            {

                traverseBHRMP(total, child, point, theta);

            }
        }

    }

};


void TSNEBHRMPNPKernel(double& total, QuadTreeBarnesHutReverseMultiPole<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
{
    glm::dvec2 R = sinkNode->centreOfMass - sourcePoint.position;
    double sq_r = R.x * R.x + R.y * R.y;
    double rS = 1.0 + sq_r;

    double D1 = 1.0 / (rS * rS);
    double D2 = -4.0 / (rS * rS * rS);
    double D3 = 24.0 / (rS * rS * rS * rS);
    total += sinkNode->totalMass / rS;

    Fastor::Tensor<double, 2> C1 =
    {
        (R.x * D1),
        (R.y * D1)
    };

    Fastor::Tensor<double, 2, 2> C2 =
    {
        {
            (D1 + R.x * R.x * D2),
            (R.x * R.y * D2)
        },
        {
            (R.y * R.x * D2),
            (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<double, 2, 2, 2> C3 =
    {
        {
            {
                ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    sinkNode->C1 += C1;
    sinkNode->C2 += C2;
    sinkNode->C3 += C3;
}

void TSNEBHRMPPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}