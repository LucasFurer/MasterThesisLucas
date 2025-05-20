#pragma once

#include "../trees/quadtree.h"
#include "../nbodysolvers/nBodySolver.h"

template <typename T>
class NBodySolverBarnesHut : public NBodySolver<T>
{
public:
    QuadTree<T> root;

    std::function<glm::vec2(float*, T, QuadTree<T>*)> kernelParticleNode;
    std::function<glm::vec2(float*, T, T)> kernelParticleParticle;

    NBodySolverBarnesHut()
    {

    }

    NBodySolverBarnesHut(std::function<glm::vec2(float*, T, QuadTree<T>*)> initKernelParticleNode, std::function<glm::vec2(float*, T, T)> initKernelParticleParticle, int initMaxChildren, float initTheta)
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
            (*forces)[i] = getBarnesHutAcc(total, &root, (*embeddedPoints)[i], this->theta);
        }
    }

    void updateTree(std::vector<T>* embeddedPoints)
    {
        root.~QuadTree();
        root = std::move(QuadTree<T>(this->maxChildren, embeddedPoints));
        this->lineSegments.clear();
        root.getLineSegments(this->lineSegments, 0, this->showLevel);
        std::vector<VertexPos2Col3> VertexPos2Col3s = LineSegment2D::LineSegmentToVertexPos2Col3(this->lineSegments);
        this->boxBuffer->createVertexBuffer(VertexPos2Col3s, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

private:
    glm::vec2 getBarnesHutAcc(float* total, QuadTree<T>* node, T particle, float theta)
    {
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 nodeDiff = particle.position - node->centreOfMass;


        if ((node->highestCorner.x - node->lowestCorner.x) / glm::length(nodeDiff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {

            acc += kernelParticleNode(total, particle, node);

        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {

                    acc += kernelParticleParticle(total, particle, (*node->allParticles)[i]);

                }
            }
        }
        else
        {
            for (QuadTree<T>* octTree : node->children)
            {

                acc += getBarnesHutAcc(total, octTree, particle, theta);

            }
        }

        return acc;
    }

};


glm::vec2 TSNEbarnesHutParticleNodeKernal(float* accumulator, EmbeddedPoint i, QuadTree<EmbeddedPoint>* j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 nodeDiff = i.position - j->centreOfMass;
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));
    *accumulator += j->totalMass * oneOverDistance;

    return -j->totalMass * oneOverDistance * oneOverDistance * nodeDiff;
}

glm::vec2 TSNEbarnesHutParticleParticleKernal(float* accumulator, EmbeddedPoint i, EmbeddedPoint j)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 diff = i.position - j.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);
    *accumulator += 1.0f * oneOverDistance;

    return -1.0f * oneOverDistance * oneOverDistance * diff;
}



glm::vec2 GRAVITYbarnesHutParticleNodeKernal(float* accumulator, Particle2D i, QuadTree<Particle2D>* j)
{
    float softening = 0.1f;

    glm::vec2 nodeDiff = i.position - j->centreOfMass;
    float parCentreDistance = glm::length(nodeDiff);

    float oneOverDistance = (1.0f / (softening + parCentreDistance));

    return -j->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * nodeDiff;
}

glm::vec2 GRAVITYbarnesHutParticleParticleKernal(float* accumulator, Particle2D i, Particle2D j)
{
    float softening = 0.1f;

    glm::vec2 diff = i.position - j.position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
}