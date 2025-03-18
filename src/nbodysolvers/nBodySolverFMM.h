#pragma once

#include "../trees/quadtreeFMM.h"

class NBodySolverFMM
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer = new Buffer();
    int showLevel = 0;

    NBodySolverFMM()
    {
    }
    
    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<EmbeddedPoint>* embeddedPoints, int maxChildren, float theta)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        QuadTreeFMM root(maxChildren, embeddedPoints);


        getFMMAcc(total, forces, &root, &root, theta);
        root.applyForces(forces);
        

        lineSegments.clear();
        root.getLineSegments(lineSegments, 0, showLevel);

        float* lineSegmentsToBuffer = LineSegment2D::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment2D));
        boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 10 * sizeof(float) * lineSegments.size(), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] lineSegmentsToBuffer;
    }
    
private:   
    void getFMMAcc(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM* passiveNode, QuadTreeFMM* activeNode, float theta)
    {
        float Lpassive = passiveNode->highestCorner.x - passiveNode->lowestCorner.x;
        float Lactive = activeNode->highestCorner.x - activeNode->lowestCorner.x;

        glm::vec2 nodeDiff = passiveNode->centreOfMass - activeNode->centreOfMass;
        float parCentreDistance = glm::length(nodeDiff);

     
        if ((Lpassive + Lactive) / parCentreDistance < theta)
        {

            float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
            *total += passiveNode->occupants.size() * activeNode->totalMass * oneOverDistance;

            passiveNode->accumulatedForce += -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;

        }
        else if (passiveNode->children.size() == 0)
        {
            for (int i = 0; i < passiveNode->occupants.size(); i++)
            {

                (*forces)[i] += getBarnesHutAcc(total, activeNode, (*passiveNode->allParticles)[passiveNode->occupants[i]], theta);

            }
        }
        else if (activeNode->children.size() == 0)
        {
            for (QuadTreeFMM* octTree : passiveNode->children)
            {
                getFMMAcc(total, forces, octTree, activeNode, theta);
            }

            //for(int i = 0; i < activeNode->occupants.size(); i++) //each[ppassive, apassive] in nodeactive do
            //{
                //TRAVERSE_PASSIVE(pactive, mactive, nodepassive)
            //}
        }
        else
        {
            for (QuadTreeFMM* octTreeFMMPassiveChild : passiveNode->children) // each childpassive in nodepassive do
            {
                for (QuadTreeFMM* octTreeFMMActiveChild : activeNode->children) // each childactive in nodeactive do
                {

                    getFMMAcc(total, forces, octTreeFMMPassiveChild, octTreeFMMActiveChild, theta);

                }
            }
        }

    }


    glm::vec2 getBarnesHutAcc(float* total, QuadTreeFMM* node, EmbeddedPoint particle, float theta)
    {
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);


        if (l / parCentreDistance < theta)
        {

            float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
            *total += node->totalMass * oneOverDistance;

            acc += -node->totalMass * oneOverDistance * oneOverDistance * nodeDiff;

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

                    acc += -1.0f * oneOverDistance * oneOverDistance * diff;

                }
            }
        }
        else
        {
            for (QuadTreeFMM* octTree : node->children)
            {
                acc += getBarnesHutAcc(total, octTree, particle, theta);
            }
        }

        return acc;
    }
    
};
