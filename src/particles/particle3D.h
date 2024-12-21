#ifndef PARTICLE3D_H
#define PARTICLE3D_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Particle3D
{
public:
    glm::vec3 position;
    glm::vec3 speed;
    glm::vec3 color;
    float mass;

    Particle3D(glm::vec3 initPosition, glm::vec3 initSpeed, glm::vec3 initColor, float initMass)
    {
        position = initPosition;
        speed = initSpeed;
        color = initColor;
        mass = initMass;
    }

    Particle3D() : position(glm::vec3(0.0f)), speed(glm::vec3(0.0f)), color(glm::vec3(0.0f))
    {
    }

    static float* Particle3DToFloat(Particle3D* particles, std::size_t particlesSize)
    {
        int particleAmount = particlesSize / sizeof(Particle3D);

        float* result = new float[6 * particleAmount];

        for (int i = 0; i < particleAmount; i++)
        {
            result[6 * i + 0] = particles[i].position.x;
            result[6 * i + 1] = particles[i].position.y;
            result[6 * i + 2] = particles[i].position.z;

            result[6 * i + 3] = particles[i].color.r;
            result[6 * i + 4] = particles[i].color.g;
            result[6 * i + 5] = particles[i].color.b;
        }

        return result;
    }
private:
};


#endif