#pragma once

#include "../trees/quadTreeBarnesHutReverseMultiPole.h"
#include "../particles/particle2D.h"
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
class NBodySolverBarnesHutReverseMultiPole : public NBodySolver<T>
{
public:
    std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    std::function<void(float*, QuadTreeBarnesHutReverseMultiPole<T>*, T)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    int maxChildren;
    float theta;

    NBodySolverBarnesHutReverseMultiPole()
    {

    }

    NBodySolverBarnesHutReverseMultiPole(std::function<void(float*, QuadTreeBarnesHutReverseMultiPole<T>*, T)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle, int initMaxChildren, float initTheta)
    {
        kernelParticleNode = initKernelParticleNode;
        kernelParticleParticle = initKernelParticleParticle;
        maxChildren = initMaxChildren;
        theta = initTheta;
    }

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));


        QuadTreeBarnesHutReverseMultiPole<T> root(maxChildren, embeddedPoints);
        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            getBarnesHutAcc(total, forces, &root, (*embeddedPoints)[i], theta);
        }
        root.applyForces(forces);
        //collapseTree(forces, &root, glm::vec2(0.0f));


        lineSegments.clear();
        root.getLineSegments(lineSegments, 0, this->showLevel);

        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
        //float* lineSegmentsToBuffer = LineSegment2D::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment2D));
        //boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 10 * sizeof(float) * lineSegments.size(), pos2DCol3D, GL_DYNAMIC_DRAW);
        //delete[] lineSegmentsToBuffer;
    }

private:
    void getBarnesHutAcc(float* total, std::vector<glm::vec2>* forces, QuadTreeBarnesHutReverseMultiPole<T>* node, T particle, float theta)
    {
        //glm::vec2 acc(0.0f);

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


    /*
    void collapseTree(std::vector<glm::vec2>* forces, QuadTreeBarnesHutReverseMultiPole<T>* node, glm::vec2 accumulatedAcc)
    {
        glm::vec2 updateDaccumulatedAcc = accumulatedAcc + node->acceleration;

        if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {

                (*forces)[i] += updateDaccumulatedAcc; // for multipole do evaluated at offset

            }
        }
        else
        {
            for (QuadTreeBarnesHutReverseMultiPole<T>* childQuadTree : node->children)
            {

                collapseTree(forces, childQuadTree, updateDaccumulatedAcc); // for multipole do evaluated at offset

            }
        }
    }
    */
};

//float softening = 1.0f;

void TSNEbarnesHutReverseMultiPoleParticleNodeKernal(float* accumulator, QuadTreeBarnesHutReverseMultiPole<EmbeddedPoint>* j, EmbeddedPoint i)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = j->centreOfMass - i.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += j->totalMass * oneOverDistance;

    //return -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}

glm::vec2 TSNEbarnesHutReverseMultiPoleParticleParticleKernal(float* accumulator, EmbeddedPoint j, EmbeddedPoint i)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);
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

    //float D0 = log(rS);
    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    float MA0 = passiveNode->totalMass;
    float MB0 = 1.0f; //activeNode->totalMass;
    Fastor::Tensor<float, 2, 2> MB2{}; // = activeNode->quadrupole;
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


    //C0 *= MA0;
    //C1 *= MA0;
    //C2 *= MA0;
    //C3 *= MA0;



    //C1 =
    //{
    //    MB0 * R.x * D1,
    //    MB0 * R.y * D1
    //};

    //C2 =
    //{
    //    {
    //        MB0 * (D1 + R.x * R.x * D2),
    //        MB0 * (R.x * R.y * D2)
    //    },
    //    {
    //        MB0 * (R.y * R.x * D2),
    //        MB0 * (D1 + R.y * R.y * D2)
    //    }
    //};


    //passiveNode->C0 += C0;
    passiveNode->C1 += C1;
    passiveNode->C2 += C2;
    passiveNode->C3 += C3;
}

glm::vec2 GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal(float* accumulator, Particle2D j, Particle2D i)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}