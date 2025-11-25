#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>
#include <boost/sort/spreadsort/float_sort.hpp>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtreeFMM.h"
#include "../../trees/cpu/nodeFMM2D.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"

template <typename T>
class NBodySolverFMM : public NBodySolver<T>
{
public:
    unsigned int treeDepth;
    std::vector<NodeFMM2D> nodes;
    std::vector<unsigned int> levelIndex;
    std::vector<unsigned int> levelSize;

    std::function<void(float&, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> kernelNN;
    std::function<void(float&, T&, QuadTreeFMM<T>*)> kernelPN;
    std::function<void(float&, QuadTreeFMM<T>*, T&)> kernelNP;
    std::function<void(float&, T&, T&)> kernelPP;

    NBodySolverFMM() {}

    NBodySolverFMM
    (
        std::function<void(float&, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> initKernelNN,
        std::function<void(float&, T&, QuadTreeFMM<T>*)> initKernelPN,
        std::function<void(float&, QuadTreeFMM<T>*, T&)> initKernelNP,
        std::function<void(float&, T&, T&)> initKernelPP,
        int initMaxChildren,
        unsigned int initTreeDepth,
        float initTheta
    )
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        initNodesSize(initTreeDepth);
        this->theta = initTheta;
    }
    
    void solveNbody(float& total, std::vector<T>& points) override
    {
        //traverseFMM(total, points, &root, &root, this->theta);

        //root.applyForces(points);

        std::vector<float> v = { 3.1f, 1.9f, -2.5f };
        boost::sort::spreadsort::float_sort(v);


        //sortMorton

        createLeafNodes(points);
    }

    void updateTree(std::vector<T>& points) override
    {
        //root = std::move(QuadTreeFMM<T>(this->maxChildren, &points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        //root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }
    
private:   
    void traverseFMM(float& total, std::vector<T>& points, QuadTreeFMM<T>* sinkNode, QuadTreeFMM<T>* sourceNode, float theta)
    {
        float Lsink = sinkNode->highestCorner.x - sinkNode->lowestCorner.x;
        float Lsource = sourceNode->highestCorner.x - sourceNode->lowestCorner.x;

        glm::vec2 diff = sinkNode->centreOfMass - sourceNode->centreOfMass;
        float dist = glm::length(diff);

     
        if ((Lsink + Lsource) / dist < theta)
        {

            kernelNN(total, sinkNode, sourceNode);

        }
        else if (sinkNode->children.size() == 0)
        {
            for (int sinkNodePointindex : sinkNode->occupants)
            {

                traverseBHMP(total, points[sinkNodePointindex], sourceNode, theta);

            }
        }
        else if (sourceNode->children.size() == 0)
        {
            for (int sourceNodeParticleIndex : sourceNode->occupants)
            {

                traverseBHRMP(total, sinkNode, points[sourceNodeParticleIndex], theta);

            }
        }
        else
        {
            for (QuadTreeFMM<T>* sinkNodeChild : sinkNode->children)
            {
                for (QuadTreeFMM<T>* sourceNodeChild : sourceNode->children)
                {

                    traverseFMM(total, points, sinkNodeChild, sourceNodeChild, theta);

                }
            }
        }

    }

    void traverseBHMP(float& total, T& sinkPoint, QuadTreeFMM<T>* sourceNode, float theta)
    {
        float l = sourceNode->highestCorner.x - sourceNode->lowestCorner.x;
        glm::vec2 diff = sinkPoint.position - sourceNode->centreOfMass;

        if (l / glm::length(diff) < theta)
        {

            kernelPN(total, sinkPoint, sourceNode);

        }
        else if (sourceNode->children.size() <= 1)
        {
            for (int i : sourceNode->occupants)
            {
                if (!glm::all(glm::equal((*sourceNode->allParticles)[i].position, sinkPoint.position)))
                {

                    kernelPP(total, sinkPoint, (*sourceNode->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTreeFMM<T>* sourceNodeChild : sourceNode->children)
            {
                traverseBHMP(total, sinkPoint, sourceNodeChild, theta);
            }
        }
    }
    
    void traverseBHRMP(float& total, QuadTreeFMM<T>* sinkNode, T& sourcePoint, float theta)
    {
        float l = sinkNode->highestCorner.x - sinkNode->lowestCorner.x;
        glm::vec2 diff = sinkNode->centreOfMass - sourcePoint.position;

        if (l / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelNP(total, sinkNode, sourcePoint);

        }
        else if (sinkNode->children.size() <= 1)
        {
            for (int i : sinkNode->occupants)
            {
                if (!glm::all(glm::equal((*sinkNode->allParticles)[i].position, sourcePoint.position)))
                {

                    kernelPP(total, (*sinkNode->allParticles)[i], sourcePoint);
                    
                }
            }
        }
        else
        {
            for (QuadTreeFMM<T>* sinkNodeChild : sinkNode->children)
            {

                traverseBHRMP(total, sinkNodeChild, sourcePoint, theta);

            }
        }

    }
    
    void initNodesSize(unsigned int initTreeDepth)
    {
        treeDepth = initTreeDepth;
        levelIndex.resize(treeDepth + 1);
        levelSize.resize(treeDepth + 1);
        levelIndex[0] = 0;
        int nodesSize = 0;
        for (int i = 0; i <= treeDepth; i++) // treeDepth = 0 means just the root
        {
            int currentLevelSize = 1;
            for (int j = 0; j < i; j++)
            {
                currentLevelSize *= 4;
            }
            levelSize[i] = currentLevelSize;
            nodesSize += currentLevelSize;

            int currentDepthStart = 0;
            for (int j = 0; j <= i; j++)
            {
                currentDepthStart += levelSize[j];
            }
            if (i + 1 < treeDepth + 1)
                levelIndex[i + 1] = currentDepthStart;
        }

        nodes.resize(nodesSize);
    }

    void createLeafNodes(std::vector<T>& points)
    {
        for (int i = 0; i < points.size(); i++)
        {
            //points[i]
        }
    }
};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



void TSNEFMMNNKernelNaive(float& total, QuadTreeFMM<TsnePoint2D>* sinkNode, QuadTreeFMM<TsnePoint2D>* sourceNode)
{
    glm::vec2 diff = sinkNode->centreOfMass - sourceNode->centreOfMass;
    float dist = glm::length(diff);

    float forceDecay = (1.0f / (1.0f + (dist * dist)));
    total += sinkNode->totalMass * sourceNode->totalMass * forceDecay;

    sinkNode->tempAccAcc += sourceNode->totalMass * forceDecay * forceDecay * diff;
}
void TSNEFMMNNKernel(float& total, QuadTreeFMM<TsnePoint2D>* sinkNode, QuadTreeFMM<TsnePoint2D>* sourceNode)
{
    glm::vec2 R = sinkNode->centreOfMass - sourceNode->centreOfMass;
    float r = glm::length(R);
    float rS = 1.0f + (r*r);
    
    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += (sinkNode->totalMass * sourceNode->totalMass) / rS;

    float MA0 = sinkNode->totalMass;
    float MB0 = sourceNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode->quadrupole;
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


    sinkNode->C1 += C1;
    sinkNode->C2 += C2;
    sinkNode->C3 += C3;
}


void TSNEFMMPNKernelNaive(float& total, TsnePoint2D& sinkPoint, QuadTreeFMM<TsnePoint2D>* sourceNode)
{
    glm::vec2 diff = sinkPoint.position - sourceNode->centreOfMass;
    float dist = glm::length(diff);

    float forceDecay = (1.0f / (1.0f + (dist * dist)));
    total += sourceNode->totalMass * forceDecay;

    sinkPoint.derivative += sourceNode->totalMass * forceDecay * forceDecay * diff;
}
void TSNEFMMPNKernel(float& total, TsnePoint2D& sinkPoint, QuadTreeFMM<TsnePoint2D>* sourceNode)
{
    glm::vec2 R = sinkPoint.position - sourceNode->centreOfMass;
    float r = glm::length(R);
    float rS = 1.0f + (r * r);

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += sourceNode->totalMass / rS;

    float MB0 = sourceNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode->quadrupole;
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

    sinkPoint.derivative += glm::vec2(C1(0), C1(1));
}


void TSNEFMMNPKernelNaive(float& total, QuadTreeFMM<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkNode->centreOfMass - sourcePoint.position; // change this
    float dist = glm::length(diff);

    float forceDecay = (1.0f / (1.0f + (dist * dist)));
    total += sinkNode->totalMass * forceDecay;

    sinkNode->tempAccAcc += forceDecay * forceDecay * diff;
}
void TSNEFMMNPKernel(float& total, QuadTreeFMM<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
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


void TSNEFMMPPKernel(float& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}






// gravity kernals ----------------------------------------------------------------------------------------------------------------------



//void GRAVITYFMMNodeNodeKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveNode->centreOfMass - activeNode->centreOfMass;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    passiveNode->tempAccAcc += -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//void GRAVITYFMMNodeNodeKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
//{
//    // prework
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    float D1 =  -1.0f / (rS * rS * rS);
//    float D2 =   3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MA0 = passiveNode->totalMass;
//    float MB0 = activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
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
//        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
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
//
//    passiveNode->C1 += C1;
//    passiveNode->C2 += C2;
//    passiveNode->C3 += C3;
//}
//
//
//glm::vec2 GRAVITYFMMParticleNodeKernalNaive(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveParticle.position - activeNode->centreOfMass;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    return -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//glm::vec2 GRAVITYFMMParticleNodeKernal(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    float D1 = -1.0f / (rS * rS * rS);
//    float D2 = 3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MB0 = activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
//    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;
//
//
//    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//    Fastor::Tensor<float, 2> C1 =
//    {
//        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//    };
//
//    return glm::vec2(C1(0), C1(1));
//}
//
//
//void GRAVITYFMMNodeParticleKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveNode->centreOfMass - activeParticle.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//void GRAVITYFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
//{
//    // prework
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
//    //glm::vec2 R = activeNode->centreOfMass - passiveNode->centreOfMass; // gadget4
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    //float D0 = log(rS);
//    float D1 = -1.0f / (rS * rS * rS);
//    float D2 = 3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MA0 = passiveNode->totalMass;
//    float MB0 = 1.0f; //activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2{}; // = activeNode->quadrupole;
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
//
//glm::vec2 GRAVITYFMMParticleParticleKernal(float* accumulator, Particle2D passiveParticle, Particle2D activeParticle)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveParticle.position - activeParticle.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = 1.0f / (distance + softening);
//    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
