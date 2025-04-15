#pragma once

#include "../trees/quadtreeFMM.h"
#include <Fastor/Fastor.h>

template <typename T>
class NBodySolverFMM
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer = new Buffer();
    int showLevel = 0;

    std::function<void(float*, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> kernelNodeNode;
    std::function<glm::vec2(float*, T, QuadTreeFMM<T>*)> kernelParticleNode;
    std::function<void(float*, QuadTreeFMM<T>*, T)> kernelNodeParticle;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    NBodySolverFMM
    (
        std::function<void(float*, QuadTreeFMM<T>*, QuadTreeFMM<T>*)> initKernelNodeNode,
        std::function<glm::vec2(float*, T, QuadTreeFMM<T>*)> initKernelParticleNode,
        std::function<void(float*, QuadTreeFMM<T>*, T)> initKernelNodeParticle,
        std::function<glm::vec2(float*, T, T)> initKernelParticleParticle
    )
    {
        kernelNodeNode = initKernelNodeNode;
        kernelParticleNode = initKernelParticleNode;
        kernelNodeParticle = initKernelNodeParticle;
        kernelParticleParticle = initKernelParticleParticle;
    }

    NBodySolverFMM() {}
    
    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints, int maxChildren, float theta)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        QuadTreeFMM root(maxChildren, embeddedPoints);


        getFMMAcc(total, forces, &root, &root, theta);
        //root.divideC();
        root.applyForces(forces);
        

        lineSegments.clear();
        root.getLineSegments(lineSegments, 0, showLevel);

        float* lineSegmentsToBuffer = LineSegment2D::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment2D));
        boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 10 * sizeof(float) * lineSegments.size(), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] lineSegmentsToBuffer;
    }
    
private:   
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
            for (int passiveNodeParticleIndex : passiveNode->occupants)
            {

                getBarnesHutAccActiveTree(total, forces, activeNode, passiveNodeParticleIndex, theta);

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
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D0 = log(rS);
    float D1 = 1.0f / (rS * rS);
    float D2 = -2.0f / (rS * rS * rS * rS);
    float D3 = 8.0f / (rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    /*
    float C0 = MB0 * (D0 + 
                      0.5f * (MB2Tilde(0,0) + MB2Tilde(1, 1)) * D1 + 
                      0.5f * (R.x*R.x*MB2Tilde(0,0) + R.x*R.y*MB2Tilde(0, 1) + R.y*R.x*MB2Tilde(1, 0) + R.y*R.y*MB2Tilde(1, 1)) * D2);
    */

    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = R.x*R.x*MB2Tilde(0, 0) + R.x*R.y*MB2Tilde(0, 1) + R.y*R.x*MB2Tilde(1, 0) + R.y*R.y*MB2Tilde(1, 1);
    float MB2TildeSum3i0 = R.x* MB2Tilde(0, 0) + R.y* MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x* MB2Tilde(1, 0) + R.y* MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 = 
    {
        MB0 * (R.x * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<float, 2, 2> C2 = 
    {
        {
            MB0 * (D1 + R.x*R.x*D2),
            MB0 * (R.x*R.y*D2)
        },
        {
            MB0 * (R.y*R.x*D2),
            MB0 * (D1 + R.y*R.y*D2)
        }
    };
    
    Fastor::Tensor<float, 2, 2, 2> C3 =
    {
        {
            {
                MB0 * ((R.x+R.x+R.x)*D2 + R.x*R.x*R.x*D3),
                MB0 * ((R.y)*D2 + R.x*R.x*R.y*D3)
            },
            {
                MB0 * ((R.y)*D2 + R.x*R.y*R.x*D3),
                MB0 * ((R.x)*D2 + R.x*R.y*R.y*D3)
            }
        },
        {
            {
                MB0 * ((R.y)*D2 + R.y*R.x*R.x*D3),
                MB0 * ((R.y)*D2 + R.y*R.x*R.y*D3)
            },
            {
                MB0 * ((R.x)*D2 + R.y*R.y*R.x*D3),
                MB0 * ((R.y+R.y+R.y)*D2 + R.y*R.y*R.y*D3)
            }
        }
    };

    //C0 *= MA0;
    C1 *= MA0;
    C2 *= MA0;
    C3 *= MA0;

    //passiveNode->C0 += C0;
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
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = activeNode->centreOfMass - passiveParticle.position;
    float r = glm::length(R) + softening;

    float g1 = -1.0f / (r * r);
    float g2 = 2.0f / (r * r * r * r);
    float g3 = -8.0f / (r * r * r * r * r * r);

    float Q0 = activeNode->totalMass;
    glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);

    glm::vec2 Q1 = activeNode->dipole;
    glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y,
        g2 * R.y * R.x, g1 + g2 * R.y * R.y);

    Fastor::Tensor<float, 2, 2, 2> D3{
                                        { { g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x, g2 * (R.y) + g3 * R.x * R.x * R.y }, { g2 * (R.y) + g3 * R.x * R.x * R.y, g2 * (R.x) + g3 * R.x * R.y * R.y } },
                                        { { g2 * (R.y) + g3 * R.y * R.x * R.x, g2 * (R.x) + g3 * R.y * R.x * R.y }, { g2 * (R.x) + g3 * R.y * R.y * R.x, g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y } }
    };
    
    Fastor::Tensor<float, 2> Q2D3 = einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(activeNode->quadrupole, D3);

    *accumulator += activeNode->totalMass * (1.0f / r);

    return -(Q0 * D1 + 0.5f * glm::vec2(Q2D3(0), Q2D3(1)));
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
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeParticle.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += passiveNode->totalMass * oneOverDistance;

    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}


glm::vec2 TSNEFMMParticleParticleKernalNaive(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (1.0f + distance);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}
glm::vec2 TSNEFMMParticleParticleKernal(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = activeParticle.position - passiveParticle.position;
    float r = glm::length(R) + softening;

    float g1 = -1.0f / (r * r);
    *accumulator += 1.0f / r;

    return -1.0f * g1 * R;
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

    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass; // dhenen
    //glm::vec2 R = activeNode->centreOfMass - passiveNode->centreOfMass; // gadget4
    float r = glm::length(R);
    float rS = r + softening;

    //float D0 = log(rS);
    float D1 =  -1.0f / (rS * rS * rS);
    float D2 =   3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    
    //float C0 = MB0 * (D0 +
    //                  0.5f * (MB2Tilde(0,0) + MB2Tilde(1, 1)) * D1 +
    //                  0.5f * (R.x*R.x*MB2Tilde(0,0) + R.x*R.y*MB2Tilde(0, 1) + R.y*R.x*MB2Tilde(1, 0) + R.y*R.y*MB2Tilde(1, 1)) * D2);
    

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

    
    //C0 *= MA0;
    //C1 *= MA0;
    //C2 *= MA0;
    //C3 *= MA0;

    //passiveNode->C0 += C0;
    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;
    
    /*
    float softening = 0.1f;

    //glm::vec2 R = j->centreOfMass - i.position;
    //float Rlen = glm::length(R);
    //float r = Rlen + softening;
    glm::vec2 R = activeNode->centreOfMass - passiveNode->centreOfMass; // gadget4
    float r = glm::length(R);
    float rS = r + softening;

    //float g0 = -log(r);
    //float g1 = -1.0f / (r * r); // should be positive actually?
    float g1 = -1.0f / (rS * rS * rS);
    //float g2 = 2.0f / (r * r * r * r); // should be positive actually?
    float g2 = 3.0f / (rS * rS * rS * rS * rS);
    //float g3 = -8.0f / (r * r * r * r * r * r); // should be positive actually?
    float g3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);





    float Q0 = activeNode->totalMass;
    //glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);
    Fastor::Tensor<float, 2> D1{ g1 * R.x, g1 * R.y };

    glm::vec2 Q1 = activeNode->dipole;
    //glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y,
    //    g2 * R.y * R.x, g1 + g2 * R.y * R.y);
    Fastor::Tensor<float, 2, 2> D2{ {g1 + g2 * R.x * R.y, g2 * R.x * R.y},
                                    {g2 * R.y * R.x, g1 + g2 * R.y * R.y} };

    Fastor::Tensor<float, 2, 2, 2> D3{
                                        { { g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x, g2 * (R.y) + g3 * R.x * R.x * R.y }, { g2 * (R.y) + g3 * R.x * R.x * R.y, g2 * (R.x) + g3 * R.x * R.y * R.y } },
                                        { { g2 * (R.y) + g3 * R.y * R.x * R.x, g2 * (R.x) + g3 * R.y * R.x * R.y }, { g2 * (R.x) + g3 * R.y * R.y * R.x, g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y } }
    };

    //Fastor::Tensor<float, 2> Q2D3 = einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(activeNode->quadrupole, D3);
    Fastor::Tensor<float, 2> Q2D3 = einsum<Fastor::Index<0, 1, 2>, Fastor::Index<0, 1>>(D3, activeNode->quadrupole);




    //C1 *= MA0;
    //C2 *= MA0;
    //C3 *= MA0;

    //passiveNode->C0 += C0;
    passiveNode->C1 +=  activeNode->totalMass * D1 + 0.5f * Q2D3;
    passiveNode->C2 += -activeNode->totalMass * D2;
    passiveNode->C3 +=  activeNode->totalMass * D3;
    */
}


glm::vec2 GRAVITYFMMParticleNodeKernalNaive(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveParticle.position - activeNode->centreOfMass;
    float distance = glm::length(diff);

    float oneOverDistance = (1.0f / (distance + softening));
    return -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
glm::vec2 GRAVITYFMMParticleNodeKernal(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = activeNode->centreOfMass - passiveParticle.position;
    float r = glm::length(R) + softening;

    float g1 = -1.0f / (r * r);
    float g2 = 2.0f / (r * r * r * r);
    float g3 = -8.0f / (r * r * r * r * r * r);

    float Q0 = activeNode->totalMass;
    glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);

    glm::vec2 Q1 = activeNode->dipole;
    glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y,
        g2 * R.y * R.x, g1 + g2 * R.y * R.y);

    Fastor::Tensor<float, 2, 2, 2> D3{
                                        { { g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x, g2 * (R.y) + g3 * R.x * R.x * R.y }, { g2 * (R.y) + g3 * R.x * R.x * R.y, g2 * (R.x) + g3 * R.x * R.y * R.y } },
                                        { { g2 * (R.y) + g3 * R.y * R.x * R.x, g2 * (R.x) + g3 * R.y * R.x * R.y }, { g2 * (R.x) + g3 * R.y * R.y * R.x, g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y } }
    };

    Fastor::Tensor<float, 2> Q2D3 = einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(activeNode->quadrupole, D3);

    *accumulator += activeNode->totalMass * (1.0f / r);

    return -(Q0 * D1 + 0.5f * glm::vec2(Q2D3(0), Q2D3(1)));
}


void GRAVITYFMMNodeParticleKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveNode->centreOfMass - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = (1.0f / (distance + softening));
    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
void GRAVITYFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeParticle.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += passiveNode->totalMass * oneOverDistance;

    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}


glm::vec2 GRAVITYFMMParticleParticleKernalNaive(float* accumulator, Particle2D passiveParticle, Particle2D activeParticle)
{
    float softening = 0.1f;
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (distance + softening);
    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
glm::vec2 GRAVITYFMMParticleParticleKernal(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = activeParticle.position - passiveParticle.position;
    float r = glm::length(R) + softening;

    float g1 = -1.0f / (r * r);
    *accumulator += 1.0f / r;

    return -1.0f * g1 * R;
}