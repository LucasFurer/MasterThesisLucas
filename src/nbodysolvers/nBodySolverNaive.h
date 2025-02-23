#pragma once

#include "../trees/quadtree.h"
#include <functional>

template <typename T>
class NBodySolverNaive
{
public:
    std::function<void(float*, std::vector<T>*, int, int, std::vector<glm::vec2>*)> kernel;

    NBodySolverNaive() 
    {

    }

    //template <typename T>
    NBodySolverNaive(std::function<void(float*, std::vector<T>*, int, int, std::vector<glm::vec2>*)> initKernel)
    {
        kernel = initKernel;
    }

    //template <typename T>
    void solveNbody(float* total, std::vector<glm::vec2>* forces, std::vector<T>* embeddedPoints)
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

                    if (kernel) {
                        kernel(total, embeddedPoints, i, j, forces);
                    }

                    /*
                    float softening = 1.0f; // should be 1.0f for t-SNE

                    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
                    float distance = glm::length(diff);

                    float oneOverDistance = 1.0f / (softening + distance);
                    *total += 1.0f * oneOverDistance;

                    (*forces)[i] += oneOverDistance * oneOverDistance * oneOverDistance * diff;
                    */
                }
            }
        }
    }

private:

};

void TSNEnaiveKernal(float* accumulator, std::vector<EmbeddedPoint>* embeddedPoints, int i, int j, std::vector<glm::vec2>* forces)
{
    float softening = 1.0f; // should be 1.0f for t-SNE

    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);
    *accumulator += 1.0f * oneOverDistance;

    (*forces)[i] += oneOverDistance * oneOverDistance * oneOverDistance * diff;
}

void GRAVITYnaiveKernal(float* accumulator, std::vector<Particle2D>* embeddedPoints, int i, int j, std::vector<glm::vec2>* forces)
{
    float softening = 0.1f; // should be 1.0f for t-SNE

    glm::vec2 diff = (*embeddedPoints)[j].position - (*embeddedPoints)[i].position;
    float distance = glm::length(diff);

    float oneOverDistance = 1.0f / (softening + distance);

    (*forces)[i] += oneOverDistance * oneOverDistance * oneOverDistance * diff;
}
