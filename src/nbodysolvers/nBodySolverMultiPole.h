#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../trees/quadtreemultipole.h"

class NBodySolverMultiPole
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer = new Buffer();
    int showLevel = 0;

    NBodySolverMultiPole()
    {
    }

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<EmbeddedPoint>* embeddedPoints, int maxChildren, float theta)
    {
        //std::cout << "start the barnes hut solver" << std::endl;

        //float timeBefore = glfwGetTime();
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));
        //std::cout << "time it took for zeroing forces array: " << glfwGetTime() - timeBefore << std::endl;

        //timeBefore = glfwGetTime();
        QuadTreeMultiPole root = QuadTreeMultiPole(maxChildren, embeddedPoints);
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
    glm::vec2 getMultiPoleAcc(float* total, QuadTreeMultiPole* node, EmbeddedPoint particle, float theta)
    {
        float softening = 1.0f; // should be 1.0f for t-SNE

        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        //glm::vec2 nodeDiff = node->centreOfMass - particle.position; // change this
        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);

        //if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            /*
            //double _r = sqrt(r2 + softening2);
            float _r = glm::length(nodeDiff) + 1.0f;

            //double prefact = -G/(_r*_r*_r)*node->m;
            float prefact = - (1.0f) / (_r * _r * _r) * node->totalMass;

            
            //double qprefact = G/(_r*_r*_r*_r*_r);
            float qprefact = 1.0f / (_r * _r * _r * _r * _r);
            //std::cout << "initial qprefact: " << qprefact << std::endl;
            //qprefact = 0.0f;
            
            //[0][0] = mxx
            //[1][1] = myy
            //[1][0] = mxy
            //[0][1] = myx
            float mxx = node->quadrupole[0][0];
            float myy = node->quadrupole[1][1];
            float mxy = node->quadrupole[1][0];
            float myx = node->quadrupole[0][1];

            //particles[pt].ax += qprefact*(dx*node->mxx + dy*node->mxy + dz*node->mxz); 
            //particles[pt].ay += qprefact*(dx*node->mxy + dy*node->myy + dz*node->myz); 
            //particles[pt].az += qprefact*(dx*node->mxz + dy*node->myz + dz*node->mzz); 
            acc.x += qprefact * (nodeDiff.x * mxx + nodeDiff.y * mxy);
            acc.y += qprefact * (nodeDiff.x * mxy + nodeDiff.y * myy);
            //double mrr     = dx*dx*node->mxx     + dy*dy*node->myy     + dz*dz*node->mzz
            //        + 2.*dx*dy*node->mxy     + 2.*dx*dz*node->mxz     + 2.*dy*dz*node->myz; 
            float mrr = (nodeDiff.x * nodeDiff.x * mxx) + (nodeDiff.y * nodeDiff.y * myy) +
                        (2.0f * nodeDiff.x * nodeDiff.y * mxy);

            //qprefact *= -5.0/(2.0*_r*_r)*mrr;
            qprefact *= (-5.0f) / (2.0f * _r * _r) * mrr; // might be wrong
            //qprefact = 0.0f;

            //particles[pt].ax += (qprefact + prefact) * dx; 
            //particles[pt].ay += (qprefact + prefact) * dy; 
            //particles[pt].az += (qprefact + prefact) * dz; 
            acc.x += (qprefact + prefact) * nodeDiff.x;
            acc.y += (qprefact + prefact) * nodeDiff.y;
            


            
            //particles[pt].ax += prefact*dx; 
            //particles[pt].ay += prefact*dy; 
            //particles[pt].az += prefact*dz; 
            acc.x += prefact * nodeDiff.x;
            acc.y += prefact * nodeDiff.y;
            

            *total += node->totalMass * (1.0f / _r);
            */

            /*
            Eigen::Tensor<float, 2> A(2, 2);
            Eigen::Tensor<float, 3> B(2, 2, 2);

            // Initialize A
            A.setValues({ {1, 2}, {3, 4} });

            // Initialize B
            B.setValues({ {{1, 2}, {3, 4}}, {{5, 6}, {7, 8}} });

            // Define the contraction pairs
            Eigen::array<Eigen::IndexPair<int>, 2> contract_dims = {
                Eigen::IndexPair<int>(0, 0),  // Contract A's 1st dim with B's 1st dim
                Eigen::IndexPair<int>(1, 1)   // Contract A's 2nd dim with B's 2nd dim
            };

            // Perform the contraction
            Eigen::Tensor<float, 1> C = A.contract(B, contract_dims);

            // Print result
            std::cout << "C: " << C << std::endl;
            */

            glm::vec2 R = node->centreOfMass - particle.position;
            float r = glm::length(R) + softening;

            //float g0 = 1.0f / r;
            float g1 = - 1.0f / (r * r * r);
            float g2 = 3.0f / (r * r * r * r * r);
            float g3 = - 15.0f / (r * r * r * r * r * r * r);

            //Eigen::Tensor<float, 1> Q0(2);
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
            




            /*
            // Define the contraction pairs
            Eigen::array<Eigen::IndexPair<int>, 2> contract_dims = {
                Eigen::IndexPair<int>(0, 0),
                Eigen::IndexPair<int>(1, 1)
            };

            Eigen::Tensor<float, 1> Q2D3 = Q2.contract(D3, contract_dims);
            */

            glm::vec2 Q2D3 = contract(Q2, D3);

            //acc += -(Q0 * D1 + Q1 * D2 + glm::vec2(Q2D3(0), Q2D3(1)));
            acc += -(Q0 * D1 + Q1 * D2 + 0.5f * Q2D3);
            //acc += -(Q0 * D1);


            *total += node->totalMass * (1.0f / r);
            /*
            float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
            *total += node->totalMass * oneOverDistance;

            acc += - node->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
            */
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {
                    glm::vec2 diff = particle.position - (*node->allParticles)[i].position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (softening + distance);
                    *total += 1.0f * oneOverDistance;

                    acc += - 1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
                }
            }
        }
        else
        {
            for (QuadTreeMultiPole* octTree : node->children)
            {
                acc += getMultiPoleAcc(total, octTree, particle, theta);
            }
        }

        return acc;
    }


    glm::vec2 contract(Eigen::Tensor<float, 2> Q2, Eigen::Tensor<float, 3> D3)
    {
        glm::vec2 result(0.0f);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    result[k] += Q2(i,j) * D3(i,j,k);
                }
            }
        }
        
        return result;
    }


};
