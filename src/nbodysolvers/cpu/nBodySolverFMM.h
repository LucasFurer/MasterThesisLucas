// this FMM is not true O(n) as it uses the BH tree construction method

#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtreeFMM.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"

template <typename T>
class NBodySolverFMM : public NBodySolver<T>
{
public:
    QuadTreeFMM<T> root;

    std::function<void(double&, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> kernelNN;
    std::function<void(double&, T&, QuadTreeFMM<T>*)> kernelPN;
    std::function<void(double&, QuadTreeFMM<T>*, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    //int FMMiter = 0;
    //int BHMPiter = 0;
    //int BHRMPiter = 0;

    //int NNiter = 0;
    //int PNiter = 0;
    //int NPiter = 0;
    //int PPiter = 0;

    NBodySolverFMM() {}

    NBodySolverFMM
    (
        std::function<void(double&, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> initKernelNN,
        std::function<void(double&, T&, QuadTreeFMM<T>*)> initKernelPN,
        std::function<void(double&, QuadTreeFMM<T>*, T&)> initKernelNP,
        std::function<void(double&, T&, T&)> initKernelPP,
        int initMaxChildren,
        double initTheta
    )
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(double& total, std::vector<T>& points) override
    {
        //FMMiter = 0;
        //BHMPiter = 0;
        //BHRMPiter = 0;

        //NNiter = 0;
        //PNiter = 0;
        //NPiter = 0;
        //PPiter = 0;

        traverseFMM(total, points, &root, &root, this->theta);

        //std::cout << "FMMiter: " << FMMiter << std::endl;
        //std::cout << "BHMPiter: " << BHMPiter << std::endl;
        //std::cout << "BHRMPiter: " << BHRMPiter << std::endl;

        //std::cout << "NNiter: " << NNiter << std::endl;
        //std::cout << "PNiter: " << PNiter << std::endl;
        //std::cout << "NPiter: " << NPiter << std::endl;
        //std::cout << "PPiter: " << PPiter << std::endl;

        root.applyForces(points);
    }

    void updateTree(std::vector<T>& points, glm::dvec2 minPos, glm::dvec2 maxPos) override
    {
        root = std::move(QuadTreeFMM<T>(this->maxChildren, &points));
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        root.getNodesBufferData(result, 0, nodeLevelToShow);
        return result;
    }

private:
    void traverseFMM(double& total, std::vector<T>& points, QuadTreeFMM<T>* sinkNode, QuadTreeFMM<T>* sourceNode, double theta)
    {
        double Lsink = sinkNode->highestCorner.x - sinkNode->lowestCorner.x;
        double Lsource = sourceNode->highestCorner.x - sourceNode->lowestCorner.x;

        glm::dvec2 diff = sinkNode->centreOfMass - sourceNode->centreOfMass;
        double dist = glm::length(diff);


        if ((Lsink + Lsource) / dist < theta)
        {
            //NNiter++;
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

    void traverseBHMP(double& total, T& sinkPoint, QuadTreeFMM<T>* sourceNode, double theta)
    {
        double l = sourceNode->highestCorner.x - sourceNode->lowestCorner.x;
        glm::dvec2 diff = sinkPoint.position - sourceNode->centreOfMass;

        if (l / glm::length(diff) < theta)
        {
            //PNiter++;
            kernelPN(total, sinkPoint, sourceNode);

        }
        else if (sourceNode->children.size() <= 1)
        {
            for (int i : sourceNode->occupants)
            {
                if (&(*sourceNode->allParticles)[i] != &sinkPoint)
                {
                    //PPiter++;
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

    void traverseBHRMP(double& total, QuadTreeFMM<T>* sinkNode, T& sourcePoint, double theta)
    {
        double l = sinkNode->highestCorner.x - sinkNode->lowestCorner.x;
        glm::dvec2 diff = sinkNode->centreOfMass - sourcePoint.position;

        if (l / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            //NPiter++;
            kernelNP(total, sinkNode, sourcePoint);

        }
        else if (sinkNode->children.size() <= 1)
        {
            for (int i : sinkNode->occupants)
            {
                if (&(*sinkNode->allParticles)[i] != &sourcePoint)
                {
                    //PPiter++;
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

};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



void TSNEFMMNNKernel(double& total, QuadTreeFMM<TsnePoint2D>* sinkNode, QuadTreeFMM<TsnePoint2D>* sourceNode)
{
    glm::dvec2 R = sinkNode->centreOfMass - sourceNode->centreOfMass;
    double sq_r = R.x * R.x + R.y * R.y;
    double rS = 1.0 + sq_r;

    double D1 = 1.0 / (rS * rS);
    double D2 = -4.0 / (rS * rS * rS);
    double D3 = 24.0 / (rS * rS * rS * rS);
    total += (sinkNode->totalMass * sourceNode->totalMass) / rS;

    double MA0 = sinkNode->totalMass;
    double MB0 = sourceNode->totalMass;
    Fastor::Tensor<double, 2, 2> MB2 = sourceNode->quadrupole;
    Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;

    // calculate the C^m
    double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<double, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<double, 2, 2> C2 =
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

    Fastor::Tensor<double, 2, 2, 2> C3 =
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


void TSNEFMMPNKernel(double& total, TsnePoint2D& sinkPoint, QuadTreeFMM<TsnePoint2D>* sourceNode)
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


void TSNEFMMNPKernel(double& total, QuadTreeFMM<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
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


void TSNEFMMPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
    double sq_dist = diff.x * diff.x + diff.y * diff.y;

    double forceDecay = 1.0 / (1.0 + sq_dist);
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}