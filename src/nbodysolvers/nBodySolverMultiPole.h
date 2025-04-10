#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../trees/quadtreemultipole.h"
#include <Fastor/Fastor.h>

template <typename T>
class NBodySolverMultiPole
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer = new Buffer();
    int showLevel = 0;

    std::function<glm::vec2(float*, T, QuadTreeMultiPole<T>*)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    NBodySolverMultiPole()
    {

    }

    NBodySolverMultiPole(std::function<glm::vec2(float*, T, QuadTreeMultiPole<T>*)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle)
    {
        kernelParticleNode = initKernelParticleNode;
        kernelParticleParticle = initKernelParticleParticle;
    }

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints, int maxChildren, float theta)
    {
        //std::cout << "start the barnes hut solver" << std::endl;

        //float timeBefore = glfwGetTime();
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));
        //std::cout << "time it took for zeroing forces array: " << glfwGetTime() - timeBefore << std::endl;

        //timeBefore = glfwGetTime();
        QuadTreeMultiPole<T> root(maxChildren, embeddedPoints);
        //std::cout << "total mass: " << root.totalMass << std::endl;
        //std::cout << "center of mass: " << glm::to_string(root.centreOfMass) << std::endl;
        //std::cout << "dipole: " << glm::to_string(root.dipole) << std::endl;
        //std::cout << "quadrupole: " << glm::to_string(root.quadrupole) << std::endl;
        //std::cout << "time it took for tree construction: " << glfwGetTime() - timeBefore << std::endl;

        //timeBefore = glfwGetTime();
        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            (*forces)[i] = getMultiPoleAcc(total, &root, (*embeddedPoints)[i], theta);
        }
        //std::cout << "time it took for force calculations: " << glfwGetTime() - timeBefore << std::endl;

        //int showLevel = 0;
        lineSegments.clear();
        root.getLineSegments(lineSegments, 0, showLevel);

        float* lineSegmentsToBuffer = LineSegment2D::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment2D));
        boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 10 * sizeof(float) * lineSegments.size(), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] lineSegmentsToBuffer;
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



glm::vec2 TSNEmultiPoleParticleNodeKernal(float* accumulator, EmbeddedPoint i, QuadTreeMultiPole<EmbeddedPoint>* j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 R = j->centreOfMass - i.position;
    float Rlen = glm::length(R);
    float r = Rlen + softening;
    
    //float g0 = -log(r);
    //float g1 = -1.0f / (r * r); // should be positive actually?
    float g1 = -1.0f / (r * r);
    //float g2 = 2.0f / (r * r * r * r); // should be positive actually?
    float g2 = 2.0f / (r * r * r * r);
    //float g3 = -8.0f / (r * r * r * r * r * r); // should be positive actually?
    float g3 = -8.0f / (r * r * r * r * r * r);


    float Q0 = j->totalMass;
    glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);

    glm::vec2 Q1 = j->dipole;
    glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y,
                             g2 * R.y * R.x, g1 + g2 * R.y * R.y);

    /*
    std::array<std::array<std::array<float, 2>, 2>, 2> D3;
    D3[0][0][0] = g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x;
    D3[1][0][0] = g2 * (R.y)             + g3 * R.y * R.x * R.x;
    D3[0][1][0] = g2 * (R.y)             + g3 * R.x * R.x * R.y;
    D3[1][1][0] = g2 * (R.x)             + g3 * R.y * R.y * R.x;

    D3[0][0][1] = g2 * (R.y)             + g3 * R.x * R.x * R.y;
    D3[1][0][1] = g2 * (R.x)             + g3 * R.y * R.x * R.y;
    D3[0][1][1] = g2 * (R.x)             + g3 * R.x * R.y * R.y;
    D3[1][1][1] = g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y;

    glm::vec2 Q2D3 = contractArray(j->quadrupole, D3);

    //return -(Q0 * D1 + Q1 * D2 + 0.5f * Q2D3);
    //return -(Q0 * D1 + 0.5f * Q2D3);
    */

    Fastor::Tensor<float, 2, 2, 2> D3{
                                        { { g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x, g2 * (R.y) + g3 * R.x * R.x * R.y }, { g2 * (R.y) + g3 * R.x * R.x * R.y, g2 * (R.x)             + g3 * R.x * R.y * R.y } },
                                        { { g2 * (R.y)             + g3 * R.y * R.x * R.x, g2 * (R.x) + g3 * R.y * R.x * R.y }, { g2 * (R.x) + g3 * R.y * R.y * R.x, g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y } }
                                     };
    /*
    Fastor::Tensor<float, 2, 2> Q2{
                                     { j->quadrupole[0][0], j->quadrupole[0][1] }, 
                                     { j->quadrupole[1][0], j->quadrupole[1][1] } 
                                  };
*/
    Fastor::Tensor<float, 2> Q2D3 = einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(j->quadrupole, D3);



    *accumulator += j->totalMass * (1.0f / r);


    return -(Q0 * D1 + 0.5f * glm::vec2(Q2D3(0), Q2D3(1)) );
}

glm::vec2 TSNEmultiPoleParticleParticleKernal(float* accumulator, EmbeddedPoint i, EmbeddedPoint j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 diff = i.position - j.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}



glm::vec2 GRAVITYmultiPoleParticleNodeKernal(float* accumulator, Particle2D i, QuadTreeMultiPole<Particle2D>* j)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 R = j->centreOfMass - i.position;
    float r = glm::length(R) + softening;

    //float g0 = 1.0f / r;
    float g1 = -1.0f / (r * r * r);
    float g2 = 3.0f / (r * r * r * r * r);
    float g3 = -15.0f / (r * r * r * r * r * r * r);

    float Q0 = j->totalMass;
    glm::vec2 D1 = glm::vec2(g1 * R.x, g1 * R.y);

    glm::vec2 Q1 = j->dipole;
    glm::mat2 D2 = glm::mat2(g1 + g2 * R.x * R.y, g2 * R.x * R.y,
        g2 * R.y * R.x, g1 + g2 * R.y * R.y);


    Eigen::Tensor<float, 2> Q2(2, 2);
    Eigen::Tensor<float, 3> D3(2, 2, 2);

    
    Q2.setValues({ {j->quadrupole(0,0), j->quadrupole(1,0)},
                   {j->quadrupole(0,1), j->quadrupole(1,1)} });
                   
    D3.setValues({ {{g2 * (R.x + R.x + R.x) + g3 * R.x * R.x * R.x, g2 * (R.y) + g3 * R.y * R.x * R.x},
                    {g2 * (R.y) + g3 * R.y * R.x * R.x,         g2 * (R.x) + g3 * R.y * R.y * R.x}},

                   {{g2 * (R.y) + g3 * R.x * R.x * R.y, g2 * (R.x) + g3 * R.y * R.y * R.x},
                    {g2 * (R.x) + g3 * R.y * R.y * R.x, g2 * (R.y + R.y + R.y) + g3 * R.y * R.y * R.y}} });
    
    glm::vec2 Q2D3 = contractTensor(Q2, D3);

    //acc += -(Q0 * D1 + Q1 * D2 + glm::vec2(Q2D3(0), Q2D3(1)));
    return -(Q0 * D1 + Q1 * D2 + 0.5f * Q2D3);
    //acc += -(Q0 * D1);
}

glm::vec2 GRAVITYmultiPoleParticleParticleKernal(float* accumulator, Particle2D i, Particle2D j)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 diff = i.position - j.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}

