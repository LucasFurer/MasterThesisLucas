#pragma once

#include "../trees/quadtree.h"

class NBodySolverBarnesHut
{
public:

    static void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<EmbeddedPoint>* embeddedPoints, int maxChildren, float theta)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));

        QuadTree root = QuadTree(maxChildren, embeddedPoints);

        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            (*forces)[i] = getBarnesHutAcc(total, &root, (*embeddedPoints)[i], theta);
        }
    }

private:
    static glm::vec2 getBarnesHutAcc(float* total, QuadTree* node, EmbeddedPoint particle, float theta)
    {
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        float parCentreDistance = glm::length(node->centreOfMass - particle.position);
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            /*
            glm::vec2 forceDirection = glm::normalize(node->centreOfMass - particle.position);
            acc += ((gravConst * node->totalMass) / (powf(parCentreDistance, 2.0f) + softening)) * forceDirection;
            */
            glm::vec2 diff = node->centreOfMass - particle.position;
            float distance = glm::length(diff);

            float Qij = 1.0f / (1.0f + distance);
            *total += Qij;

            acc += Qij * (1.0f / (1.0f + distance)) * diff;
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {


                    /*
                    float parParDistance = glm::length((*node->allParticles)[i].position - particle.position); 
                    
                    glm::vec2 forceDirection = glm::normalize(node->allParticles[i].position - particle.position);
                    acc += ((gravConst * OctTree::allParticles[i].mass) / (powf(parParDistance, 2.0f) + softening)) * forceDirection;
                    */


                    glm::vec2 diff = (*node->allParticles)[i].position - particle.position;
                    float distance = glm::length(diff);

                    float Qij = 1.0f / (1.0f + distance);
                    *total += Qij;

                    acc += Qij * (1.0f / (1.0f + distance)) * diff;
                }
            }
        }
        else
        {
            for (QuadTree* octTree : node->children)
            {
                acc += getBarnesHutAcc(total, octTree, particle, theta);
            }
        }

        return acc;
    }




};
