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

    std::function<void(float&, QuadTreeBarnesHutReverseMultiPole<T>*, T&)> kernelNP;
    std::function<void(float&, T&, T&)> kernelPP;

    NBodySolverBHRMP() {}

    NBodySolverBHRMP(std::function<void(float&, QuadTreeBarnesHutReverseMultiPole<T>*, T&)> initKernelNP, std::function<void(float&, T&, T&)> initKernelPP, int initMaxChildren, float initTheta)
    {
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(float& total, std::vector<T>& points) override
    {
        total = 0.0f;

        for (int i = 0; i < points.size(); i++)
        {
            traverseBHRMP(total, &root, points[i], this->theta);
        }

        root.applyForces(points);
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
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
    void traverseBHRMP(float& total, QuadTreeBarnesHutReverseMultiPole<T>* node, T point, float theta)
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
            for (QuadTreeBarnesHutReverseMultiPole<T>* child : node->children)
            {

                traverseBHRMP(total, child, point, theta);

            }
        }

    }

};


void TSNEBHRMPNPKernel(float& total, QuadTreeBarnesHutReverseMultiPole<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
{
    glm::vec2 R = sinkNode->centreOfMass - sourcePoint.position;
    float r = glm::length(R);
    float rS = 1.0f + (r * r);

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += sinkNode->totalMass / rS;

    Fastor::Tensor<float, 2> C1 =
    {
        (R.x * D1),
        (R.y * D1)
    };

    Fastor::Tensor<float, 2, 2> C2 =
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

    Fastor::Tensor<float, 2, 2, 2> C3 =
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

void TSNEBHRMPPPKernel(float& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float oneOverDistance = 1.0f / (1.0f + (dist * dist));
    total += oneOverDistance;

    sinkPoint.derivative += oneOverDistance * oneOverDistance * diff;
}



//void GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal(float* accumulator, QuadTreeBarnesHutReverseMultiPole<Particle2D>* passiveNode, Particle2D activeParticle)
//{
//    // prework
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    float D1 = -1.0f / (rS * rS * rS);
//    float D2 = 3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MA0 = passiveNode->totalMass;
//    float MB0 = 1.0f; //activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2{};
//    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;
//
//    // calculate the C^m
//    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//    Fastor::Tensor<float, 2> C1 =
//    {
//        MB0 * (R.x * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//    };
//
//    Fastor::Tensor<float, 2, 2> C2 =
//    {
//        {
//            MB0 * (D1 + R.x * R.x * D2),
//            MB0 * (R.x * R.y * D2)
//        },
//        {
//            MB0 * (R.y * R.x * D2),
//            MB0 * (D1 + R.y * R.y * D2)
//        }
//    };
//
//    Fastor::Tensor<float, 2, 2, 2> C3 =
//    {
//        {
//            {
//                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
//                MB0 * ((R.y) * D2             + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.y) * D2             + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
//                MB0 * ((R.x) * D2             + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
//            }
//        },
//        {
//            {
//                MB0 * ((R.y) * D2             + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
//                MB0 * ((R.x) * D2             + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.x) * D2             + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
//                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
//            }
//        }
//    };
//
//    passiveNode->C1 += C1;
//    passiveNode->C2 += C2;
//    passiveNode->C3 += C3;
//}
//
//glm::vec2 GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal(float* accumulator, Particle2D j, Particle2D i)
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