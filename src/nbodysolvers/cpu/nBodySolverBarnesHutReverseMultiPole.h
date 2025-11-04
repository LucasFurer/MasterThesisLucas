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
#include "../../particles/Particle2D.h"

template <typename T>
class NBodySolverBarnesHutReverseMultiPole : public NBodySolver<T>
{
public:
    QuadTreeBarnesHutReverseMultiPole<T> root;

    std::function<void(float*, QuadTreeBarnesHutReverseMultiPole<T>*, T)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    NBodySolverBarnesHutReverseMultiPole()
    {

    }

    NBodySolverBarnesHutReverseMultiPole(std::function<void(float*, QuadTreeBarnesHutReverseMultiPole<T>*, T)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle, int initMaxChildren, float initTheta)
    {
        kernelParticleNode = initKernelParticleNode;
        kernelParticleParticle = initKernelParticleParticle;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        //updateTree(embeddedPoints);
        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            getBarnesHutAcc(total, forces, &root, (*embeddedPoints)[i], this->theta);
        }
        root.applyForces(forces);
    }

    void updateTree(std::vector<T>* embeddedPoints)
    {
        root = std::move(QuadTreeBarnesHutReverseMultiPole<T>(this->maxChildren, embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    void getBarnesHutAcc(float* total, std::vector<glm::vec2>* forces, QuadTreeBarnesHutReverseMultiPole<T>* node, T particle, float theta)
    {
        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = particle.position - node->centreOfMass;



        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelParticleNode(total, node, particle);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    (*forces)[i] += kernelParticleParticle(total, (*node->allParticles)[i], particle);

                }
            }
        }
        else
        {
            for (QuadTreeBarnesHutReverseMultiPole<T>* childQuadTree : node->children)
            {

                getBarnesHutAcc(total, forces, childQuadTree, particle, theta);

            }
        }

    }

};

//float softening = 1.0f;

void TSNEbarnesHutReverseMultiPoleParticleNodeKernal(float* accumulator, QuadTreeBarnesHutReverseMultiPole<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    // prework
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    *accumulator += passiveNode->totalMass / rS;

    float MA0 = passiveNode->totalMass;
    float MB0 = 1.0f; //activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2{};
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<float, 2, 2> C2 =
    {
        {
            MB0 * (D1 + R.x * R.x * D2),
            MB0 * (R.x * R.y * D2)
        },
        {
            MB0 * (R.y * R.x * D2),
            MB0 * (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 =
    {
        {
            {
                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;
}

glm::vec2 TSNEbarnesHutReverseMultiPoleParticleParticleKernal(float* accumulator, EmbeddedPoint j, EmbeddedPoint i)
{
    float softening = 1.0f;

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + (distance * distance));
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}



void GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal(float* accumulator, QuadTreeBarnesHutReverseMultiPole<Particle2D>* passiveNode, Particle2D activeParticle)
{
    // prework
    float softening = 0.1f;

    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = 1.0f; //activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2{};
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<float, 2, 2> C2 =
    {
        {
            MB0 * (D1 + R.x * R.x * D2),
            MB0 * (R.x * R.y * D2)
        },
        {
            MB0 * (R.y * R.x * D2),
            MB0 * (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 =
    {
        {
            {
                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                MB0 * ((R.y) * D2             + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                MB0 * ((R.y) * D2             + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                MB0 * ((R.x) * D2             + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                MB0 * ((R.y) * D2             + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                MB0 * ((R.x) * D2             + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                MB0 * ((R.x) * D2             + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;
}

glm::vec2 GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal(float* accumulator, Particle2D j, Particle2D i)
{
    float softening = 0.1f;

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}