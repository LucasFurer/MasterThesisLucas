#pragma once

#include <glm/glm.hpp>
#include <cstddef>

class Particle2D
{
public:
    glm::vec2 position;
    glm::vec2 speed;
    glm::vec3 color;
    float mass;

    Particle2D(glm::vec2 initPosition, glm::vec2 initSpeed, glm::vec3 initColor, float initMass)
    {
        position = initPosition;
        speed = initSpeed;
        color = initColor;
        mass = initMass;
    }

    Particle2D() : position(glm::vec2(0.0f)), speed(glm::vec2(0.0f)), color(glm::vec3(1.0f))
    {
    }

    static float* Particle2DToFloat(Particle2D* particles, std::size_t particlesSize)
    {
        int particleAmount = particlesSize / sizeof(Particle2D);

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