#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class EmbeddedPoint
{
public:
    glm::vec2 position;
    int label;

    EmbeddedPoint(glm::vec2 initPosition, int initLabel)
    {
        position = initPosition;
        label = initLabel;
    }

    EmbeddedPoint() : position(glm::vec2(0.0f)), label(0)
    {
    }

    static float* Particle2DToFloat(EmbeddedPoint* particles, std::size_t particlesSize)
    {
        int particleAmount = particlesSize / sizeof(EmbeddedPoint);

        float* result = new float[5 * particleAmount];

        for (int i = 0; i < particleAmount; i++)
        {
            result[5 * i + 0] = particles[i].position.x;
            result[5 * i + 1] = particles[i].position.y;

            result[5 * i + 2] = particles[i].color.r;
            result[5 * i + 3] = particles[i].color.g;
            result[5 * i + 4] = particles[i].color.b;
        }

        return result;
    }
private:
};