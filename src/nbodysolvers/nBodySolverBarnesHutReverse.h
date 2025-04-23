#pragma once

#include "../trees/quadTreeBarnesHutReverse.h"
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
class NBodySolverBarnesHutReverse : public NBodySolver<T>
{
public:
    //std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    std::function<glm::vec2(float*, T, QuadTreeBarnesHutReverse<T>*)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    //int maxChildren;
    //float theta;

    NBodySolverBarnesHutReverse()
    {

    }

    NBodySolverBarnesHutReverse(std::function<glm::vec2(float*, T, QuadTreeBarnesHutReverse<T>*)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle, int initMaxChildren, float initTheta)
    {
        kernelParticleNode = initKernelParticleNode;
        kernelParticleParticle = initKernelParticleParticle;
        this->maxChildren = initMaxChildren;
        this->theta = initTheta;
    }

    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));


        QuadTreeBarnesHutReverse<T> root(this->maxChildren, embeddedPoints);
        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            getBarnesHutAcc(total, forces, &root, (*embeddedPoints)[i], this->theta);
        }
        collapseTree(forces, &root, glm::vec2(0.0f));


        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
        //float* lineSegmentsToBuffer = LineSegment2D::LineSegmentToFloat(lineSegments.data(), lineSegments.size() * sizeof(LineSegment2D));
        //boxBuffer->createVertexBuffer(lineSegmentsToBuffer, 10 * sizeof(float) * lineSegments.size(), pos2DCol3D, GL_DYNAMIC_DRAW);
        //delete[] lineSegmentsToBuffer;
    }

private:
    void getBarnesHutAcc(float* total, std::vector<glm::vec2>* forces, QuadTreeBarnesHutReverse<T>* node, T particle, float theta)
    {
        //glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = particle.position - node->centreOfMass;


        
        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            node->acceleration += kernelParticleNode(total, particle, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    (*forces)[i] += kernelParticleParticle(total, particle, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTreeBarnesHutReverse<T>* childQuadTree : node->children)
            {

                getBarnesHutAcc(total, forces, childQuadTree, particle, theta);

            }
        }
        
    }



    void collapseTree(std::vector<glm::vec2>* forces, QuadTreeBarnesHutReverse<T>* node, glm::vec2 accumulatedAcc)
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
            for (QuadTreeBarnesHutReverse<T>* childQuadTree : node->children)
            {

                collapseTree(forces, childQuadTree, updateDaccumulatedAcc); // for multipole do evaluated at offset

            }
        }
    }

};

//float softening = 1.0f;

glm::vec2 TSNEbarnesHutReverseParticleNodeKernal(float* accumulator, EmbeddedPoint i, QuadTreeBarnesHutReverse<EmbeddedPoint>* j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = j->centreOfMass - i.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += j->totalMass * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * nodeDiff;
}

glm::vec2 TSNEbarnesHutReverseParticleParticleKernal(float* accumulator, EmbeddedPoint i, EmbeddedPoint j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}



glm::vec2 GRAVITYbarnesHutReverseParticleNodeKernal(float* accumulator, Particle2D i, QuadTreeBarnesHutReverse<Particle2D>* j)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = j->centreOfMass - i.position; // change this
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    
    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
}

glm::vec2 GRAVITYbarnesHutReverseParticleParticleKernal(float* accumulator, Particle2D i, Particle2D j)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 diff = j.position - i.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    return -i.mass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}