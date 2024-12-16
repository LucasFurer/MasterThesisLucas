#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Particle
{
public:
    // the program ID
    glm::vec3 position;
    glm::vec3 speed;
    glm::vec3 color;
    float mass;

    // constructor reads and builds the shader
    Particle(glm::vec3 initPosition, glm::vec3 initSpeed, glm::vec3 initColor, float initMass)
    {
        position = initPosition;
        speed = initSpeed;
        color = initColor;
        mass = initMass;
    }

    Particle() : position(glm::vec3(0.0f)), speed(glm::vec3(0.0f)), color(glm::vec3(0.0f))
    {
    }

    static float* ParticleToFloat(Particle* particles, std::size_t particlesSize)
    {
        int particleAmount = particlesSize / sizeof(Particle);

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