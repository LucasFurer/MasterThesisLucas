#pragma once

#include "../trees/quadtreeFMM.h"

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

    NBodySolverFMM()
    {
    }
    
    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints, int maxChildren, float theta)
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

            (*forces)[particleIndex] += TSNEFMMParticleNodeKernal(total, particle, node);

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



void TSNEFMMNodeNodeKernal(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeNode->centreOfMass;
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
    *accumulator += passiveNode->occupants.size() * activeNode->totalMass * oneOverDistance;

    passiveNode->dphi += -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;
}


glm::vec2 TSNEFMMParticleNodeKernal(float* accumulator, EmbeddedPoint passiveParticle, QuadTreeFMM<EmbeddedPoint>* activeNode)
{
    glm::vec2 nodeDiff = passiveParticle.position - activeNode->centreOfMass; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (1.0f + parCentreDistance));
    *accumulator += activeNode->totalMass * oneOverDistance;

    return -activeNode->totalMass * oneOverDistance * oneOverDistance * nodeDiff;
}


void TSNEFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<EmbeddedPoint>* passiveNode, EmbeddedPoint activeParticle)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = passiveNode->centreOfMass - activeParticle.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += passiveNode->totalMass * oneOverDistance;

    passiveNode->dphi += -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}


glm::vec2 TSNEFMMParticleParticleKernal(float* accumulator, EmbeddedPoint passiveParticle, EmbeddedPoint activeParticle)
{
    glm::vec2 diff = passiveParticle.position - activeParticle.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (1.0f + distance);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}