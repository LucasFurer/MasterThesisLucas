#pragma once

#include "../trees/quadtree.h"

class NBodySolverBarnesHut
{
public:

    static void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<EmbeddedPoint>* embeddedPoints, int maxChildren, float theta)
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
            (*forces)[i] = getBarnesHutAcc(total, &root, (*embeddedPoints)[i], theta);
        }
        //std::cout << "time it took for force calculations: " << glfwGetTime() - timeBefore << std::endl;
    }

private:
    static glm::vec2 getBarnesHutAcc(float* total, QuadTree* node, EmbeddedPoint particle, float theta)
    {
        glm::vec2 acc(0.0f);

        float l = node->highestCorner.x - node->lowestCorner.x;
        glm::vec2 cubeCentre = ((node->highestCorner + node->lowestCorner) / 2.0f);

        glm::vec2 nodeDiff = node->centreOfMass - particle.position;
        float parCentreDistance = glm::length(nodeDiff);
        if ((node->highestCorner.x - node->lowestCorner.x) / parCentreDistance < theta && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
        {
            float Qij = 1.0f / (1.0f + parCentreDistance);
            *total += Qij;

            acc += Qij * (1.0f / (1.0f + parCentreDistance)) * nodeDiff;
        }
        else if (node->children.size() <= 1)
        {
            for (int i : node->occupants)
            {
                if (!glm::all(glm::equal((*node->allParticles)[i].position, particle.position)))
                {
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
