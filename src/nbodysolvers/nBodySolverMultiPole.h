#pragma once

//#include <Eigen/Core>
//#include <unsupported/Eigen/CXX11/Tensor>
#include "../trees/quadtreemultipole.h"
#include <Fastor/Fastor.h>
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
class NBodySolverMultiPole : public NBodySolver<T>
{
public:
    //std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    QuadTreeMultiPole<T> root;

    std::function<glm::vec2(float*, T, QuadTreeMultiPole<T>*)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    //int maxChildren;
    //float theta;

    NBodySolverMultiPole()
    {

    }

    NBodySolverMultiPole(std::function<glm::vec2(float*, T, QuadTreeMultiPole<T>*)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle, int initMaxChildren, float initTheta)
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
            (*forces)[i] = getMultiPoleAcc(total, &root, (*embeddedPoints)[i], this->theta);
        }
    }

    void updateTree(std::vector<T>* embeddedPoints)
    {
        root = std::move(QuadTreeMultiPole<T>(this->maxChildren, embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    glm::vec2 getMultiPoleAcc(float* total, QuadTreeMultiPole<T>* node, T particle, float theta)
    {
        //float softening = 1.0f; // should be 1.0f for t-SNE

        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        //glm::vec2 nodeDiff = node->centreOfMass - particle.position; // change this
        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);

        //if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            acc += kernelParticleNode(total, particle, node);

            /*
            glm::vec2 R = node->centreOfMass - particle.position;
            float r = glm::length(R) + softening;

            //float g0 = 1.0f / r;
            float g1 = - 1.0f / (r * r * r);
            float g2 = 3.0f / (r * r * r * r * r);
            float g3 = - 15.0f / (r * r * r * r * r * r * r);

            float Q0 = node->totalMass;
            glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);

            glm::vec2 Q1 = node->dipole; 
            glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y, 
                                     g2 * R.y * R.x, g1 + g2 * R.y * R.y);


            Eigen::Tensor<float, 2> Q2(2, 2);
            Eigen::Tensor<float, 3> D3(2, 2, 2);

            Q2.setValues({ {node->quadrupole[0][0], node->quadrupole[1][0]}, 
                           {node->quadrupole[0][1], node->quadrupole[1][1]} });
            
            D3.setValues({ {{g2*(R.x+R.x+R.x) + g3*R.x*R.x*R.x, g2*(R.y) + g3*R.y*R.x*R.x}, 
                            {g2*(R.y) + g3*R.y*R.x*R.x,         g2*(R.x) + g3*R.y*R.y*R.x}}, 
                
                           {{g2*(R.y) + g3*R.x*R.x*R.y, g2*(R.x) + g3*R.y*R.y*R.x}, 
                            {g2*(R.x) + g3*R.y*R.y*R.x, g2*(R.y+R.y+R.y) + g3*R.y*R.y*R.y}} });
            
            glm::vec2 Q2D3 = contract(Q2, D3);

            //acc += -(Q0 * D1 + Q1 * D2 + glm::vec2(Q2D3(0), Q2D3(1)));
            acc += -(Q0 * D1 + Q1 * D2 + 0.5f * Q2D3);
            //acc += -(Q0 * D1);

            *total += node->totalMass * (1.0f / r);
            * */
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {
                    acc += kernelParticleParticle(total, particle, (*node->allParticles)[i]);
                    /*
                    float softening = 1.0f;

                    glm::vec2 diff = particle.position - (*node->allParticles)[i].position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (softening + distance);
                    *total += 1.0f * oneOverDistance;

                    acc += - 1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
                    */
                }
            }
        }
        else
        {
            for (QuadTreeMultiPole<T>* octTree : node->children)
            {
                acc += getMultiPoleAcc(total, octTree, particle, theta);
            }
        }

        return acc;
    }

};









glm::vec2 contractTensor(Eigen::Tensor<float, 2> Q2, Eigen::Tensor<float, 3> D3)
{
    glm::vec2 result(0.0f);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                result[k] += Q2(i, j) * D3(i, j, k);
            }
        }
    }

    return result;
}

glm::vec2 contractArray(glm::mat2 Q2, std::array<std::array<std::array<float, 2>, 2>, 2>& D3)
{
    glm::vec2 result(0.0f);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                result[k] += Q2[i][j] * D3[i][j][k];
            }
        }
    }

    return result;
}



glm::vec2 TSNEmultiPoleParticleNodeKernal(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeMultiPole<EmbeddedPoint>* activeNode)
{
    float softening = 1.0f;

    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS);
    float D2 = 2.0f / (rS * rS * rS * rS);
    float D3 = -8.0f / (rS * rS * rS * rS * rS * rS);
    *accumulator += activeNode->totalMass / rS;

    //float MA0 = 1.0f;//passiveNode->totalMass;
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

glm::vec2 TSNEmultiPoleParticleParticleKernal(float* accumulator, EmbeddedPoint i, EmbeddedPoint j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = i.position - j.position;
    float r = glm::length(R);
    float rS = r + softening;

    float oneOverDistance = 1.0f / (rS);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * R;
}



glm::vec2 GRAVITYmultiPoleParticleNodeKernal(float* accumulator, Particle2D passiveParticle, QuadTreeMultiPole<Particle2D>* activeNode)
{
    float softening = 0.1f;

    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
    float r = glm::length(R);
    float rS = r + softening;

    float D1 = -1.0f / (rS * rS * rS);
    float D2 = 3.0f / (rS * rS * rS * rS * rS);
    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

    //float MA0 = 1.0f;//passiveNode->totalMass;
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

glm::vec2 GRAVITYmultiPoleParticleParticleKernal(float* accumulator, Particle2D i, Particle2D j)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 R = i.position - j.position;
    float r = glm::length(R);
    float rS = r + softening;


    float oneOverDistance = 1.0f / (rS);

    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * R;
}

