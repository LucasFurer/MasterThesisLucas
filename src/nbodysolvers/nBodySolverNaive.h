#pragma once

#include "../trees/quadtree.h"

class NBodySolverNaive
{
public:

    static void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<EmbeddedPoint>* embeddedPoints)
    {
        std::fill(forces->begin(), forces->end(), glm::vec2(0.0f, 0.0f));
        *total = 0.0f;

        for (int i = 0; i < embeddedPoints->size(); i++)
        {
            for (int j = 0; j < embeddedPoints->size(); j++)
            {
                if (i != j)//might be useless
                {
                    /*
                    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
                    float distance = glm::length(diff);

                    float Qij = 1.0f / (1.0f + distance);
                    *total += Qij;

                    (*forces)[i] += Qij * (1.0f / (1.0f + distance)) * glm::normalize(diff);
                    */

                    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (1.0f + distance);
                    *total += 1.0f * oneOverDistance;

                    (*forces)[i] += oneOverDistance * oneOverDistance * oneOverDistance * diff;
                }
            }
        }
    }

private:
};
