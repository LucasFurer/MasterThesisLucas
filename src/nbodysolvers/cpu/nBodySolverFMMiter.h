#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadTreeNodeFMMiter.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/Particle2D.h"

template <typename T>
struct IntPair
{
    QuadTreeNodeFMMiter<T>* A;
    QuadTreeNodeFMMiter<T>* B;

    IntPair(QuadTreeNodeFMMiter<T>* initA, QuadTreeNodeFMMiter<T>* initB)
    {
        A = initA;
        B = initB;
    }

    static bool intPairNotIn(std::vector<IntPair>& vec, IntPair& intpair)
    {
        for (int i = 0; i < vec.size(); i++)
        {
            if ((vec[i].A == intpair.A && vec[i].B == intpair.B) || (vec[i].A == intpair.B && vec[i].B == intpair.A)) // maybe also add || (vec[i].A == intpair.B && vec[i].B == intpair.A)
            {
                return false;
            }
        }

        return true;
    }
};

template <typename T>
class NBodySolverFMMiter : public NBodySolver<T>
{
public:
    QuadTreeNodeFMMiter<T> root;

    std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> kernelInteractNodeNode;
    std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> kernelInteractNodeParticle;
    std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> kernelInteractParticleNode;
    std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> kernelInteractParticleParticle;

    std::vector<IntPair<T>> interactionList;

    NBodySolverFMMiter
    (
        std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> initKernelInteractNodeNode,
        std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> initKernelInteractNodeParticle,
        std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> initKernelInteractParticleNode,
        std::function<void(float*, QuadTreeNodeFMMiter<T>*, QuadTreeNodeFMMiter<T>*)> initKernelInteractParticleParticle,
        int initMaxChildren,
        float initTheta
    )
    {
        kernelInteractNodeNode = initKernelInteractNodeNode;
        kernelInteractNodeParticle = initKernelInteractNodeParticle;
        kernelInteractParticleNode = initKernelInteractParticleNode;
        kernelInteractParticleParticle = initKernelInteractParticleParticle;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    NBodySolverFMMiter() {}

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints) override
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        //updateTree(embeddedPoints);
        getFMMAcc(total, forces, this->theta, embeddedPoints);
        root.divideC();
        root.applyForces(forces);
    }

    void updateTree(std::vector<T>* embeddedPoints) override
    {
        root = std::move(QuadTreeNodeFMMiter<T>(this->maxChildren, *embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    void getFMMAcc(float* total, std::vector<glm::vec2>* forces, float theta, std::vector<T>* embeddedPoints)
    {
        interactionList.clear();
        interactionList.push_back(IntPair<T>(&root, &root));
        
        while (interactionList.size() != 0)
        {
            IntPair currentPair = interactionList.back();
            interactionList.pop_back();

            float LA = currentPair.A->highestCorner.x - currentPair.A->lowestCorner.x;
            float LB = currentPair.B->highestCorner.x - currentPair.B->lowestCorner.x;

            glm::vec2 nodeDiff = currentPair.A->centreOfMass - currentPair.B->centreOfMass;
            float distanceNodeNode = glm::length(nodeDiff);


            if (currentPair.A->children.size() != 0)
            {

                if (currentPair.B->children.size() != 0) // Node Node interaction
                {
                    if ((LA + LB) / distanceNodeNode < theta)
                    {
                        kernelInteractNodeNode(total, currentPair.A, currentPair.B);
                    }
                    else
                    {
                        //std::vector<IntPair<T>> addedPairs; // slow symmetric method
                        //addedPairs.reserve(16);
                        //for (QuadTreeNodeFMMiter<T>* quadTreeNodeFMMiterA : currentPair.A->children)
                        //{
                        //    for (QuadTreeNodeFMMiter<T>* quadTreeNodeFMMiterB : currentPair.B->children)
                        //    {
                        //        IntPair<T> newIntPair(quadTreeNodeFMMiterA, quadTreeNodeFMMiterB);
                        //        if (IntPair<T>::intPairNotIn(addedPairs, newIntPair))
                        //        {
                        //            addedPairs.push_back(newIntPair);
                        //            interactionList.push_back(newIntPair);
                        //        }
                        //        //interactionList.push_back(newIntPair);
                        //    }
                        //}

                        if (currentPair.A == currentPair.B)
                        {
                            for (int i = 0; i < currentPair.A->children.size(); i++) // fast symmetric method
                            {
                                for (int j = i; j < currentPair.B->children.size(); j++)
                                {
                                    interactionList.push_back(IntPair<T>(currentPair.A->children[i], currentPair.B->children[j]));
                                }
                            }
                        }
                        else
                        {
                            for (int i = 0; i < currentPair.A->children.size(); i++) // fast symmetric method
                            {
                                for (int j = 0; j < currentPair.B->children.size(); j++)
                                {
                                    interactionList.push_back(IntPair<T>(currentPair.A->children[i], currentPair.B->children[j]));
                                }
                            }
                        }


                    }
                }
                else // Node Particle interaction
                {

                    if ((LA + LB) / distanceNodeNode < theta)
                    {
                        kernelInteractNodeParticle(total, currentPair.A, currentPair.B);
                        kernelInteractParticleNode(total, currentPair.B, currentPair.A);
                    }
                    else
                    {
                        for (QuadTreeNodeFMMiter<T>* quadTreeNodeFMMiterA : currentPair.A->children)
                        {
                            interactionList.push_back(IntPair<T>(quadTreeNodeFMMiterA, currentPair.B));
                        }
                    }
                }
            }
            else
            {
                if (currentPair.B->children.size() != 0) // Particle Node interaction
                {
                    if ((LA + LB) / distanceNodeNode < theta)
                    {
                        kernelInteractParticleNode(total, currentPair.A, currentPair.B);
                        kernelInteractNodeParticle(total, currentPair.B, currentPair.A);
                    }
                    else
                    {
                        for (QuadTreeNodeFMMiter<T>* quadTreeNodeFMMiterB : currentPair.B->children)
                        {
                            interactionList.push_back(IntPair<T>(currentPair.A, quadTreeNodeFMMiterB));
                        }
                    }
                }
                else // Particle Particle interaction
                {
                    if (distanceNodeNode != 0.0f) // nodes with the same position will not be calculated since it will result in nonsense, this also prevents self interaction
                    {
                        kernelInteractParticleParticle(total, currentPair.A, currentPair.B);
                        kernelInteractParticleParticle(total, currentPair.B, currentPair.A);
                    }
                }
            }
        }
    }
};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------

void TSNEFMMiterInteractionKernalNodeNode(float* accumulator, QuadTreeNodeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeNodeFMMiter<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    *accumulator += (passiveNode->totalMass * activeNode->totalMass) / rS;

    float MA0 = passiveNode->totalMass;
    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 = MA0 * Fastor::Tensor<float, 2>
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };



    Fastor::Tensor<float, 2, 2> C2 = MA0 * Fastor::Tensor<float, 2, 2>
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

    Fastor::Tensor<float, 2, 2, 2> C3 = MA0 * Fastor::Tensor<float, 2, 2, 2>
    {
        {
            {
                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)              // i = 0, j = 0, k = 1
            },
            {
                MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3),             // i = 0, j = 1, k = 0
                MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)              // i = 0, j = 1, k = 1
            }
        },
        {
            {
                MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3),             // i = 1, j = 0, k = 0
                MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)              // i = 1, j = 0, k = 1
            },
            {
                MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3),             // i = 1, j = 1, k = 0
                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;


    // ---------------------------------------------------------------------
    
    R = -R;

    *accumulator += (activeNode->totalMass * passiveNode->totalMass) / rS;

    MA0 = activeNode->totalMass;
    MB0 = passiveNode->totalMass;
    MB2 = passiveNode->quadrupole;
    MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1other = MA0 * Fastor::Tensor<float, 2>
    {
        MB0* (R.x* (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0* (R.y* (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    activeNode->C1 += C1other;
    activeNode->C2 += C2;
    activeNode->C3 += -C3;
}


void TSNEFMMiterInteractionKernalNodeParticle(float* accumulator, QuadTreeNodeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeNodeFMMiter<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    *accumulator += passiveNode->totalMass / rS;

    float MA0 = passiveNode->totalMass;

    Fastor::Tensor<float, 2> C1 = MA0 * Fastor::Tensor<float, 2>
    {
        R.x * D1,
        R.y * D1
    };

    Fastor::Tensor<float, 2, 2> C2 = MA0 * Fastor::Tensor<float, 2, 2>
    {
        {
            D1 + R.x * R.x * D2,
            R.x * R.y * D2
        },
        {
            R.y * R.x * D2,
            D1 + R.y * R.y * D2
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 = MA0 * Fastor::Tensor<float, 2, 2, 2>
    {
        {
            {
                (R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3, // i = 0, j = 0, k = 0
                (R.y) * D2 + R.x * R.x * R.y * D3  // i = 0, j = 0, k = 1
            },
            {
                (R.y) * D2 + R.x * R.y * R.x * D3, // i = 0, j = 1, k = 0
                (R.x) * D2 + R.x * R.y * R.y * D3  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                (R.y) * D2 + R.y * R.x * R.x * D3, // i = 1, j = 0, k = 0
                (R.x) * D2 + R.y * R.x * R.y * D3  // i = 1, j = 0, k = 1
            },
            {
                (R.x) * D2 + R.y * R.y * R.x * D3, // i = 1, j = 1, k = 0
                (R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3  // i = 1, j = 1, k = 1
            }
        }
    };


    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;
}


void TSNEFMMiterInteractionKernalParticleNode(float* accumulator, QuadTreeNodeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeNodeFMMiter<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    *accumulator += activeNode->totalMass / rS;

    float MA0 = passiveNode->totalMass;
    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;


    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 = MA0 * Fastor::Tensor<float, 2>
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    passiveNode->C1 += C1;
}


void TSNEFMMiterInteractionKernalParticleParticle(float* accumulator, QuadTreeNodeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeNodeFMMiter<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);

    *accumulator += 1.0f / rS;


    Fastor::Tensor<float, 2> C1 = // massPassive * Fastor::Tensor<float, 2>
    {
        R.x * D1,
        R.y * D1
    };


    passiveNode->C1 += C1;
}


void TSNEFMMiterInteractionKernal(float* accumulator, QuadTreeNodeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeNodeFMMiter<EmbeddedPoint>* activeNode, std::vector<glm::vec2>* forces)
{
    // prework
    float softening = 1.0f;

    if (passiveNode->id == -1)
    {
        if (activeNode->id == -1)
        {
            glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
            float r = glm::length(R);
            float rS = r + softening;

            float D1 = -1.0f / (rS * rS);
            float D2 = 2.0f / (rS * rS * rS * rS);
            float D3 = -8.0f / (rS * rS * rS * rS * rS * rS);
            *accumulator += (passiveNode->totalMass * activeNode->totalMass) / rS;

            float MA0 = passiveNode->totalMass;
            float MB0 = activeNode->totalMass;
            Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
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
        else
        {
            glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
            float r = glm::length(R);
            float rS = r + softening;

            float D1 = -1.0f / (rS * rS);
            float D2 = 2.0f / (rS * rS * rS * rS);
            float D3 = -8.0f / (rS * rS * rS * rS * rS * rS);
            *accumulator += passiveNode->totalMass / rS;

            float MA0 = passiveNode->totalMass;

            Fastor::Tensor<float, 2> C1 =
            {
                R.x * D1,
                R.y * D1
            };

            Fastor::Tensor<float, 2, 2> C2 =
            {
                {
                    D1 + R.x * R.x * D2,
                    R.x * R.y * D2
                },
                {
                    R.y * R.x * D2,
                    D1 + R.y * R.y * D2
                }
            };

            Fastor::Tensor<float, 2, 2, 2> C3 =
            {
                {
                    {
                        (R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3, // i = 0, j = 0, k = 0
                        (R.y) * D2 + R.x * R.x * R.y * D3  // i = 0, j = 0, k = 1
                    },
                    {
                        (R.y) * D2 + R.x * R.y * R.x * D3, // i = 0, j = 1, k = 0
                        (R.x) * D2 + R.x * R.y * R.y * D3  // i = 0, j = 1, k = 1
                    }
                },
                {
                    {
                        (R.y) * D2 + R.y * R.x * R.x * D3, // i = 1, j = 0, k = 0
                        (R.x) * D2 + R.y * R.x * R.y * D3  // i = 1, j = 0, k = 1
                    },
                    {
                        (R.x) * D2 + R.y * R.y * R.x * D3, // i = 1, j = 1, k = 0
                        (R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3  // i = 1, j = 1, k = 1
                    }
                }
            };

            passiveNode->C1 += C1;
            passiveNode->C2 += C2;
            passiveNode->C3 += C3;
        }
    }
    else
    {
        if (activeNode->id == -1)
        {
            glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
            float r = glm::length(R);
            float rS = r + softening;

            float D1 = -1.0f / (rS * rS);
            float D2 = 2.0f / (rS * rS * rS * rS);
            float D3 = -8.0f / (rS * rS * rS * rS * rS * rS);
            *accumulator += activeNode->totalMass / rS;

            float MB0 = activeNode->totalMass;
            Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
            Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;


            float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
            float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
            float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
            float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

            Fastor::Tensor<float, 2> C1 =
            {
                MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
                MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
            };

            (*forces)[passiveNode->id] += glm::vec2(C1(0), C1(1));
        }
        else
        {
            glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
            float r = glm::length(R);
            float rS = r + softening;

            float D1 = -1.0f / (rS * rS);

            *accumulator += 1.0f / rS;

            (*forces)[passiveNode->id] += D1 * R;
        }
    }
}



// gravity kernals ----------------------------------------------------------------------------------------------------------------------


/*
void GRAVITYFMMiterNodeNodeKernal(float* accumulator, QuadTreeFMMiter<Particle2D>* passiveNode, QuadTreeFMMiter<Particle2D>* activeNode)
{
    // prework
    float softening = 0.1f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
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


glm::vec2 GRAVITYFMMiterParticleNodeKernal(float* accumulator, Particle2D passiveParticle, QuadTreeFMMiter<Particle2D>* activeNode)
{
    float softening = 0.1f;

    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;


    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    return glm::vec2(C1(0), C1(1));
}


void GRAVITYFMMiterNodeParticleKernal(float* accumulator, QuadTreeFMMiter<Particle2D>* passiveNode, Particle2D activeParticle)
{
    // prework
    float softening = 0.1f;

    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
    //glm::vec2 R = activeNode->centreOfMass - passiveNode->centreOfMass; // gadget4
    float r = glm::length(R);
    float rS = r + softening;

    //float D0 = log(rS);
    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = 1.0f; //activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2{}; // = activeNode->quadrupole;
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


glm::vec2 GRAVITYFMMiterParticleParticleKernal(float* accumulator, Particle2D passiveParticle, Particle2D activeParticle)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (distance + softening);
    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
*/