#pragma once

#include "../trees/cpu/quadtreeFMM.h"
#include "../nbodysolvers/cpu/nBodySolver.h"

template <typename T>
class NBodySolverFMM : public NBodySolver<T>
{
public:
    //std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    QuadTreeFMM<T> root;

    std::function<void(float*, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> kernelNodeNode;
    std::function<glm::vec2(float*, T, QuadTreeFMM<T>*)> kernelParticleNode;
    std::function<void(float*, QuadTreeFMM<T>*, T)> kernelNodeParticle;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    //int maxChildren;
    //float theta;

    NBodySolverFMM
    (
        std::function<void(float*, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> initKernelNodeNode,
        std::function<glm::vec2(float*, T, QuadTreeFMM<T>*)> initKernelParticleNode,
        std::function<void(float*, QuadTreeFMM<T>*, T)> initKernelNodeParticle,
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

    NBodySolverFMM() {}
    
    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints) override
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        //updateTree(embeddedPoints);
        getFMMAcc(total, forces, &root, &root, this->theta);
        //root.divideC();
        root.applyForces(forces);
    }

    void updateTree(std::vector<T>* embeddedPoints)
    {
        root = std::move(QuadTreeFMM<T>(this->maxChildren, embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }
    
private:   
    void getFMMAccIt()
    {

    }

    void getFMMAcc(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM<T>* passiveNode, QuadTreeFMM<T>* activeNode, float theta)
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

            
            //if (activeNode->children.size() == 0) // naive
            //{
            //    for (int ip : passiveNode->occupants)
            //    {
            //        for (int ia : activeNode->occupants)
            //        {
            //            (*forces)[ip] += kernelParticleParticle(total, (*passiveNode->allParticles)[ip], (*activeNode->allParticles)[ia]);
            //        }
            //    }
            //}
            //else // split
            //{
            //    for (QuadTreeFMM<T>* child : activeNode->children)
            //    {
            //        getFMMAcc(total, forces, passiveNode, child, theta);
            //    }
            //}
            
            
            for (int passivenodeparticleindex : passiveNode->occupants)
            {

                getBarnesHutAccActiveTree(total, forces, activeNode, passivenodeparticleindex, theta);

            }
            
        }
        else if (activeNode->children.size() == 0)
        {


            //if (passiveNode->children.size() == 0) // naive
            //{
            //    for (int ip : passiveNode->occupants)
            //    {
            //        for (int ia : activeNode->occupants)
            //        {
            //            (*forces)[ip] += kernelParticleParticle(total, (*passiveNode->allParticles)[ip], (*activeNode->allParticles)[ia]);
            //        }
            //    }
            //}
            //else // split
            //{
            //    for (QuadTreeFMM<T>* child : passiveNode->children)
            //    {
            //        getFMMAcc(total, forces, child, activeNode, theta);
            //    }
            //}
            
            for (int activeNodeParticleIndex : activeNode->occupants)
            {

                getBarnesHutAccPassiveTree(total, forces, passiveNode, activeNodeParticleIndex, theta);

            }
            
        }
        else
        {
            for (QuadTreeFMM<T>* octTreeFMMPassiveChild : passiveNode->children) // each childpassive in nodepassive do
            {
                for (QuadTreeFMM<T>* octTreeFMMActiveChild : activeNode->children) // each childactive in nodeactive do
                {

                    getFMMAcc(total, forces, octTreeFMMPassiveChild, octTreeFMMActiveChild, theta);

                }
            }
        }

    }

    void getBarnesHutAccActiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM<T>* node, int particleIndex, float theta)
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
            for (QuadTreeFMM<T>* octTree : node->children)
            {
                getBarnesHutAccActiveTree(total, forces, octTree, particleIndex, theta);
            }
        }
    }
    
    void getBarnesHutAccPassiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM<T>* node, int particleIndex, float theta)
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
            for (QuadTreeFMM<T>* childQuadTree : node->children)
            {

                getBarnesHutAccPassiveTree(total, forces, childQuadTree, particleIndex, theta);

            }
        }

    }
    
};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



void TSNEFMMNodeNodeKernalNaive(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeNode->centreOfMass;
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
    *accumulator += passiveNode->occupants.size() * activeNode->totalMass * oneOverDistance;


    passiveNode->tempAccAcc += -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;
}
void TSNEFMMNodeNodeKernal(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    // prework
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r*r) + softening;
    
    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    //float D2 = (4.0f * r) / (rS * rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    //float D3 = -(-4.0f + 28*r*r) / (rS * rS * rS * rS * rS * rS);
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


glm::vec2 TSNEFMMParticleNodeKernalNaive(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    glm::vec2 nodeDiff = passiveParticle.position - activeNode->centreOfMass; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
    *accumulator += activeNode->totalMass * oneOverDistance;

    return -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;
}
glm::vec2 TSNEFMMParticleNodeKernal(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
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


void TSNEFMMNodeParticleKernalNaive(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeParticle.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += passiveNode->totalMass * oneOverDistance;

    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}
void TSNEFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    // prework
    float softening = 1.0f;

    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position;
    float r = glm::length(R);
    float rS = (r * r) + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 4.0f / (rS * rS * rS);
    //float D2 = (4.0f * r) / (rS * rS * rS * rS);
    float D3 = -24.0f / (rS * rS * rS * rS);
    //float D3 = -(-4.0f + 28*r*r) / (rS * rS * rS * rS * rS * rS);
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


glm::vec2 TSNEFMMParticleParticleKernal(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    float softening = 1.0f;
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / ((distance * distance) + softening);
    *accumulator += 1.0f * oneOverDistance;
    return -1.0f * oneOverDistance * oneOverDistance * diff;
}






// gravity kernals ----------------------------------------------------------------------------------------------------------------------



void GRAVITYFMMNodeNodeKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveNode->centreOfMass - activeNode->centreOfMass;
    float distance = glm::length(diff);

    float oneOverDistance = (1.0f / (distance + softening));
    passiveNode->tempAccAcc += -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
void GRAVITYFMMNodeNodeKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
{
    // prework
    float softening = 0.1f;

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 =  -1.0f / (rS * rS * rS);
    float D2 =   3.0f / (rS * rS * rS * rS * rS);
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


glm::vec2 GRAVITYFMMParticleNodeKernalNaive(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveParticle.position - activeNode->centreOfMass;
    float distance = glm::length(diff);

    float oneOverDistance = (1.0f / (distance + softening));
    return -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
glm::vec2 GRAVITYFMMParticleNodeKernal(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
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


void GRAVITYFMMNodeParticleKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveNode->centreOfMass - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = (1.0f / (distance + softening));
    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
void GRAVITYFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
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


glm::vec2 GRAVITYFMMParticleParticleKernal(float* accumulator, Particle2D passiveParticle, Particle2D activeParticle)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (distance + softening);
    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
