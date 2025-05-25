#pragma once

#include "../trees/quadtreeFMMiter.h"
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
struct IntPair
{
    IntPair(QuadTreeFMMiter<T>* initA, QuadTreeFMMiter<T>* initB) : A(initA), B(initB)
    {

    }

    QuadTreeFMMiter<T>* A;
    QuadTreeFMMiter<T>* B;
};

template <typename T>
class NBodySolverFMMiter : public NBodySolver<T>
{
public:
    QuadTreeFMMiter<T> root;

    std::function<void(float*, QuadTreeFMMiter<T>*, QuadTreeFMMiter<T>*)> kernelNodeNode;
    std::function<glm::vec2(float*, T, QuadTreeFMMiter<T>*)> kernelParticleNode;
    std::function<void(float*, QuadTreeFMMiter<T>*, T)> kernelNodeParticle;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    NBodySolverFMMiter
    (
        std::function<void(float*, QuadTreeFMMiter<T>*, QuadTreeFMMiter<T>*)> initKernelNodeNode,
        std::function<glm::vec2(float*, T, QuadTreeFMMiter<T>*)> initKernelParticleNode,
        std::function<void(float*, QuadTreeFMMiter<T>*, T)> initKernelNodeParticle,
        std::function<glm::vec2(float*, T, T)> initKernelParticleParticle,
        int initMaxChildren,
        float initTheta
    )
    {
        kernelNodeNode = initKernelNodeNode;
        kernelParticleNode = initKernelParticleNode;
        kernelNodeParticle = initKernelNodeParticle;
        kernelParticleParticle = initKernelParticleParticle;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    NBodySolverFMMiter() {}

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints) override
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        //updateTree(embeddedPoints);
        //getFMMAcc(total, forces, &root, &root, this->theta);
        getFMMAcc(total, forces, this->theta);
        //root.divideC();
        root.applyForces(forces);
    }

    void updateTree(std::vector<T>* embeddedPoints)
    {
        root = std::move(QuadTreeFMMiter<T>(this->maxChildren, embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    void getFMMAcc(float* total, std::vector<glm::vec2>* forces, float theta)
    {
        std::queue<IntPair<T>> interactionList;
        interactionList.push(IntPair<T>(&root, &root));

        while (!interactionList.empty())
        {
            IntPair<T> currentPair = interactionList.front();
            interactionList.pop();

            float LA = currentPair.A->highestCorner.x - currentPair.A->lowestCorner.x;
            float LB = currentPair.B->highestCorner.x - currentPair.B->lowestCorner.x;

            glm::vec2 nodeDiff = currentPair.A->centreOfMass - currentPair.B->centreOfMass;
            float parCentreDistance = glm::length(nodeDiff);


            if ((LA + LB) / parCentreDistance < theta)
            {

                kernelNodeNode(total, currentPair.A, currentPair.B);
                kernelNodeNode(total, currentPair.B , currentPair.A);

            }
            else if (currentPair.A->children.size() == 0)
            {

                for (int particleIndexA : currentPair.A->occupants)
                {

                    getBarnesHutAccActiveTree(total, forces, currentPair.B, particleIndexA, theta);
                    getBarnesHutAccPassiveTree(total, forces, currentPair.B, particleIndexA, theta);

                }

            }
            else if (currentPair.B->children.size() == 0)
            {

                for (int particleIndexB : currentPair.B->occupants)
                {

                    getBarnesHutAccPassiveTree(total, forces, currentPair.A, particleIndexB, theta);
                    getBarnesHutAccActiveTree(total, forces, currentPair.A, particleIndexB, theta);

                }

            }
            else
            {
                for (QuadTreeFMMiter<T>* childA : currentPair.A->children) // each childpassive in nodepassive do
                {
                    for (QuadTreeFMMiter<T>* childB : currentPair.B->children) // each childactive in nodeactive do
                    {

                        interactionList.push(IntPair<T>(childA, childB));
                        //getFMMAcc(total, forces, childA, childB, theta);
                        //getFMMAcc(total, forces, childB, childA, theta);

                    }
                }
            }
        }
    }

    //(float* total, std::vector<glm::vec2>* forces, QuadTreeFMMiter<T>* node, int particleIndex, float theta)
    void getBarnesHutAccActiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMMiter<T>* node, int particleIndex, float theta)
    {
        T particle = (*node->allParticles)[particleIndex];

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);


        if (l / parCentreDistance < theta)
        {

            (*forces)[particleIndex] += kernelParticleNode(total, particle, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    (*forces)[particleIndex] += kernelParticleParticle(total, particle, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTreeFMMiter<T>* octTree : node->children)
            {
                getBarnesHutAccActiveTree(total, forces, octTree, particleIndex, theta);
            }
        }
    }









    void getFMMAccOLD(float* total, std::vector<glm::vec2>* forces, QuadTreeFMMiter<T>* passiveNode, QuadTreeFMMiter<T>* activeNode, float theta)
    {
        float Lpassive = passiveNode->highestCorner.x - passiveNode->lowestCorner.x;
        float Lactive = activeNode->highestCorner.x - activeNode->lowestCorner.x;

        glm::vec2 nodeDiff = passiveNode->centreOfMass - activeNode->centreOfMass;
        float parCentreDistance = glm::length(nodeDiff);


        if ((Lpassive + Lactive) / parCentreDistance < theta)
        {

            kernelNodeNode(total, passiveNode, activeNode);

        }
        else if (passiveNode->children.size() == 0)
        {

            for (int passivenodeparticleindex : passiveNode->occupants)
            {

                getBarnesHutAccActiveTree(total, forces, activeNode, passivenodeparticleindex, theta);

            }

        }
        else if (activeNode->children.size() == 0)
        {

            for (int activeNodeParticleIndex : activeNode->occupants)
            {

                getBarnesHutAccPassiveTree(total, forces, passiveNode, activeNodeParticleIndex, theta);

            }

        }
        else
        {
            for (QuadTreeFMMiter<T>* octTreeFMMPassiveChild : passiveNode->children) // each childpassive in nodepassive do
            {
                for (QuadTreeFMMiter<T>* octTreeFMMActiveChild : activeNode->children) // each childactive in nodeactive do
                {

                    getFMMAcc(total, forces, octTreeFMMPassiveChild, octTreeFMMActiveChild, theta);

                }
            }
        }

    }

    void getBarnesHutAccActiveTreeOLD(float* total, std::vector<glm::vec2>* forces, QuadTreeFMMiter<T>* node, int particleIndex, float theta)
    {
        T particle = (*node->allParticles)[particleIndex];

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);


        if (l / parCentreDistance < theta)
        {

            (*forces)[particleIndex] += kernelParticleNode(total, particle, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    (*forces)[particleIndex] += kernelParticleParticle(total, particle, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTreeFMMiter<T>* octTree : node->children)
            {
                getBarnesHutAccActiveTree(total, forces, octTree, particleIndex, theta);
            }
        }
    }

    void getBarnesHutAccPassiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMMiter<T>* node, int particleIndex, float theta)
    {
        T particle = (*node->allParticles)[particleIndex];

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = particle.position - node->centreOfMass;


        if (l / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            kernelNodeParticle(total, node, particle);

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
            for (QuadTreeFMMiter<T>* childQuadTree : node->children)
            {

                getBarnesHutAccPassiveTree(total, forces, childQuadTree, particleIndex, theta);

            }
        }

    }

};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



void TSNEFMMiterNodeNodeKernal(float* accumulator, QuadTreeFMMiter<EmbeddedPoint>* passiveNode, QuadTreeFMMiter<EmbeddedPoint>* activeNode)
{
    // prework
    float softening = 1.0f;

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


glm::vec2 TSNEFMMiterParticleNodeKernal(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeFMMiter<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
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

    return glm::vec2(C1(0), C1(1));
}


void TSNEFMMiterNodeParticleKernal(float* accumulator, QuadTreeFMMiter<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    // prework
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 2.0f / (rS * rS * rS * rS);
    float D3 = -8.0f / (rS * rS * rS * rS * rS * rS);
    *accumulator += passiveNode->totalMass / rS;

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


glm::vec2 TSNEFMMiterParticleParticleKernal(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    float softening = 1.0f;
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (distance + softening);
    *accumulator += 1.0f * oneOverDistance;
    return -1.0f * oneOverDistance * oneOverDistance * diff;
}



// gravity kernals ----------------------------------------------------------------------------------------------------------------------



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
