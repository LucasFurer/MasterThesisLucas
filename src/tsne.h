#ifndef TSNE_H
#define TSNE_H

//#include "octtree.h"
//#include "ffthelper.h"
//#include "common.h"
//#include <vector>
#include "buffer.h"

class TSNE
{
public:
	Particle2D* particles;
	std::size_t particlesSize;
	Buffer* particlesBuffer;

	TSNE()
	{

	}

	TSNE(int particleAmount)
	{
		particles = new Particle2D[particleAmount];
		particlesSize = particleAmount * sizeof(Particle2D);

        float sizeParam = 300.0f;
        for (int i = 0; i < particleAmount; i++)
        {
            glm::vec2 pos = glm::vec2(0.0f);
            glm::vec2 speed = glm::vec2(0.0f);
            glm::vec3 col = glm::vec3(0.0f);

            float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            while (powf(randX, 2.0f) + powf(randY, 2.0f) + powf(randZ, 2.0f) > 1.0f)
            {
                randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }

            pos = glm::vec2(
                powf(sizeParam * randX, 1.0f),
                powf(sizeParam * randY, 1.0f)
            );


            speed = glm::vec2(
                (((float)rand() / RAND_MAX) - 0.5f) / 30.0f,
                (((float)rand() / RAND_MAX) - 0.5f) / 30.0f
            );


            col = glm::vec3(
                randX * 0.5f + 0.5f,
                randY * 0.5f + 0.5f,
                randZ * 0.5f + 0.5f
            );

            particles[i] = Particle2D(pos, speed, col, 1.0f);
        }
        float* particlesToBuffer = Particle2D::Particle2DToFloat(particles, particlesSize);
        particlesBuffer = new Buffer(particlesToBuffer, 5 * sizeof(float) * (particlesSize / sizeof(Particle3D)), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] particlesToBuffer;
	}
	
	~TSNE()
	{
		delete[] particles;
	}

private:

};

#endif