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
        QuadTreeMultiPole root = QuadTreeMultiPole(maxChildren, embeddedPoints);
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
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        //glm::vec2 nodeDiff = node->centreOfMass - particle.position; // change this
        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);

        //if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            //std::cout << "-----------------------------------------" << std::endl;

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
            //particles[pt].ax += qprefact*(dx*node->mxx + dy*node->mxy + dz*node->mxz); 
            //particles[pt].ay += qprefact*(dx*node->mxy + dy*node->myy + dz*node->myz); 
            //particles[pt].az += qprefact*(dx*node->mxz + dy*node->myz + dz*node->mzz); 
            acc.x += qprefact * (nodeDiff.x*node->quadrupole[0][0] + nodeDiff.y*node->quadrupole[1][0]);
            acc.y += qprefact * (nodeDiff.x*node->quadrupole[1][0] + nodeDiff.y*node->quadrupole[1][1]);
            //double mrr     = dx*dx*node->mxx     + dy*dy*node->myy     + dz*dz*node->mzz
            //        + 2.*dx*dy*node->mxy     + 2.*dx*dz*node->mxz     + 2.*dy*dz*node->myz; 
            float mrr = (nodeDiff.x * nodeDiff.x * node->quadrupole[0][0]) + (nodeDiff.y * nodeDiff.y * node->quadrupole[1][1]) +
                        (2.0f * nodeDiff.x * nodeDiff.y * node->quadrupole[1][0]);

            //qprefact *= -5.0/(2.0*_r*_r)*mrr;
            qprefact *= (-5.0f) / (2.0f * _r * _r) * mrr; // might be wrong

            
            //std::cout << "current accx: " << acc.x << std::endl;
            //std::cout << "current accy: " << acc.y << std::endl;
            //std::cout << "current mrr: " << mrr << std::endl;
            //std::cout << "current quad: " << glm::to_string(node->quadrupole) << std::endl;
            //std::cout << "current prefact: " << prefact << std::endl;
            //std::cout << "current qprefact: " << qprefact << std::endl;

            //particles[pt].ax += (qprefact + prefact) * dx; 
            //particles[pt].ay += (qprefact + prefact) * dy; 
            //particles[pt].az += (qprefact + prefact) * dz; 
            acc.x += (qprefact + prefact) * nodeDiff.x;
            acc.y += (qprefact + prefact) * nodeDiff.y;
            


            /*
            //particles[pt].ax += prefact*dx; 
            //particles[pt].ay += prefact*dy; 
            //particles[pt].az += prefact*dz; 
            acc.x += prefact * nodeDiff.x;
            acc.y += prefact * nodeDiff.y;
            */

            *total += node->totalMass * (1.0f / _r);


            //float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
            //*total += node->totalMass * oneOverDistance;

            //acc += - node->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {
                    glm::vec2 diff = particle.position - (*node->allParticles)[i].position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (1.0f + distance);
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




};
