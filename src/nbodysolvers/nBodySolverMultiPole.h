#pragma once

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
        QuadTree root = QuadTree(maxChildren, embeddedPoints);
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
    glm::vec2 getMultiPoleAcc(float* total, QuadTree* node, EmbeddedPoint particle, float theta)
    {
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        //glm::vec2 nodeDiff = node->centreOfMass - particle.position; // change this
        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);

        //if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            nodeDiff;

            //double _r = sqrt(r2 + softening2);
            //double prefact = -G/(_r*_r*_r)*node->m;

            //double qprefact = G/(_r*_r*_r*_r*_r);
            //particles[pt].ax += qprefact*(dx*node->mxx + dy*node->mxy + dz*node->mxz); 
            //particles[pt].ay += qprefact*(dx*node->mxy + dy*node->myy + dz*node->myz); 
            //particles[pt].az += qprefact*(dx*node->mxz + dy*node->myz + dz*node->mzz); 
            //double mrr     = dx*dx*node->mxx     + dy*dy*node->myy     + dz*dz*node->mzz
            //        + 2.*dx*dy*node->mxy     + 2.*dx*dz*node->mxz     + 2.*dy*dz*node->myz; 
            //qprefact *= -5.0/(2.0*_r*_r)*mrr;
            //particles[pt].ax += (qprefact + prefact) * dx; 
            //particles[pt].ay += (qprefact + prefact) * dy; 
            //particles[pt].az += (qprefact + prefact) * dz; 










            float Qij = node->totalMass * (1.0f / (1.0f + parCentreDistance));
            *total += Qij;

            //acc += -Qij * (1.0f / (1.0f + parCentreDistance)) * glm::normalize(nodeDiff);
            acc += -Qij * (1.0f / (1.0f + parCentreDistance)) * (nodeDiff / parCentreDistance);
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {
                    //glm::vec2 diff = (*node->allParticles)[i].position - particle.position;
                    glm::vec2 diff = particle.position - (*node->allParticles)[i].position;
                    float distance = glm::length(diff);

                    float Qij = 1.0f / (1.0f + distance);
                    *total += Qij;

                    //acc += -Qij * (1.0f / (1.0f + distance)) * glm::normalize(diff);
                    acc += -Qij * (1.0f / (1.0f + distance)) * (diff / distance);
                }
            }
        }
        else
        {
            for (QuadTree* octTree : node->children)
            {
                acc += getMultiPoleAcc(total, octTree, particle, theta);
            }
        }

        return acc;
    }




};
