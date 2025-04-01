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

            passiveNode->dphi += -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;

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
            for (QuadTreeFMM* octTreeFMMPassiveChild : passiveNode->children) // each childpassive in nodepassive do
            {
                for (QuadTreeFMM* octTreeFMMActiveChild : activeNode->children) // each childactive in nodeactive do
                {

                    getFMMAcc(total, forces, octTreeFMMPassiveChild, octTreeFMMActiveChild, theta);

                }
            }
        }

    }


    void getBarnesHutAccActiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM* node, int particleIndex, float theta)
    {
        EmbeddedPoint particle = (*node->allParticles)[particleIndex];

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        glm::vec2 nodeDiff = particle.position - node->centreOfMass; // change this
        float parCentreDistance = glm::length(nodeDiff);


        if (l / parCentreDistance < theta)
        {

            float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
            *total += node->totalMass * oneOverDistance;

            (*forces)[particleIndex] += -node->totalMass * oneOverDistance * oneOverDistance * nodeDiff;

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

                    (*forces)[particleIndex] += -1.0f * oneOverDistance * oneOverDistance * diff;

                }
            }
        }
        else
        {
            for (QuadTreeFMM* octTree : node->children)
            {
                getBarnesHutAccActiveTree(total, forces, octTree, particleIndex, theta);
            }
        }
    }
    
    void getBarnesHutAccPassiveTree(float* total, std::vector<glm::vec2>* forces, QuadTreeFMM* node, int particleIndex, float theta)
    {
        EmbeddedPoint particle = (*node->allParticles)[particleIndex];

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = particle.position - node->centreOfMass;



        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            //node->dphi += kernelParticleNode(total, particle, node);
            float softening = 1.0f; // should be 1.0f for t-SNE

            glm::vec2 nodeDiff = node->centreOfMass - particle.position; // change this
            float parCentreDistance = glm::length(nodeDiff);

            float oneOverDistance = (1.0f / (softening + parCentreDistance));
            *total += node->totalMass * oneOverDistance;

            glm::vec2 result = -1.0f * oneOverDistance * oneOverDistance * nodeDiff;

            node->dphi += result;

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    //(*forces)[i] += kernelParticleParticle(total, particle, (*node->allParticles)[i]);
                    float softening = 1.0f; // should be 1.0f for t-SNE

                    glm::vec2 diff = (*node->allParticles)[i].position - particle.position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (softening + distance);
                    *total += 1.0f * oneOverDistance;

                    glm::vec2 result = -1.0f * oneOverDistance * oneOverDistance * diff;

                    (*forces)[i] += result;

                }
            }
        }
        else
        {
            for (QuadTreeFMM* childQuadTree : node->children)
            {

                getBarnesHutAccPassiveTree(total, forces, childQuadTree, particleIndex, theta);

            }
        }

    }
    
};
