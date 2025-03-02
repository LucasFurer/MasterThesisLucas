#pragma once
#include <vector>
#include "particles/particle3D.h"
#include "buffer.h"
#include "nbodysolvers/nBodySolverNaive.h"

class GravitySim
{
public:
	std::vector<Particle2D> particles;
	std::vector<glm::vec2> accelerations;
	std::vector<glm::vec2> accelerationsErrorTest;
	Buffer* particlesBuffer;

    float stepSize = 0.1f;
    float timeStepsPerSec = 100000.0f;
    float lastTimeUpdated = 0.0f;

    NBodySolverNaive<Particle2D> nBodySolverNaive;
    NBodySolverBarnesHut<Particle2D> nBodySolverBarnesHut;
    NBodySolverMultiPole<Particle2D> nBodySolverMultiPole;
    NBodySolverFMM nBodySolverFMM;

	GravitySim(int particleAmount)
	{
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        nBodySolverNaive = NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
        nBodySolverBarnesHut = NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal);
        nBodySolverMultiPole = NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal);

        srand(1952731);
        float sizeParam = 200.0f;
        for (int i = 0; i < particleAmount; i++)
        {
            float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;


            while (powf(randX, 2.0f) + powf(randY, 2.0f) > 1.0f)
            {
                randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }


            glm::vec2 pos = glm::vec2(
                sizeParam * randX,
                sizeParam * randY
            );

            float initVelMag = 15.0f;
            glm::vec2 vel = glm::vec2(
                initVelMag * ((float)rand() / RAND_MAX) - (initVelMag/2.0f),
                initVelMag * ((float)rand() / RAND_MAX) - (initVelMag/2.0f)
            );

            particles[i] = Particle2D(pos, vel, glm::vec3(1.0f), 1.0f);
        }

        particlesBuffer = new Buffer(particles.data(), particles.size(), pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
	}

    ~GravitySim()
    {
        // delete embeddedBuffer?
    }

    void cleanup()
    {
        particlesBuffer->cleanup();
    }


    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            
            //std::tuple errorResult = checkError();
            //std::cout << "difference in error: " << std::get<0>(errorResult) << std::endl;
            //std::cout << "ratio of error:      " << std::get<1>(errorResult) << std::endl;
            //std::cout << "------------------------------------------------------" << std::endl;


            float noAccumulator = 0.0f;
            //nBodySolverNaive.solveNbody(&noAccumulator, &accelerations, &particles);
            nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f); // keep theta between 0.0 (off) and 1.0 (can be higher) 0.3 gives no artifacts
            //nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
            //nBodySolverFMM.solveNbody(&QijTotal, &repulsForce, &embeddedPoints, 10, 0.8f);

            /*
            std::cout << "=================================================================" << std::endl;
            for (int i = 0; i < accelerations.size(); i++)
            {
                std::cout << "acceleration of particle " << i << ": " << glm::to_string(accelerations[i]) << std::endl;
                std::cout << "speed of particle        " << i << ": " << glm::to_string(particles[i].speed) << std::endl;
                std::cout << "pos of particle          " << i << ": " << glm::to_string(particles[i].position) << std::endl;
                std::cout << "--------------------------------------------------------------"  << std::endl;
            }
            */

            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];
            
            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBufferNew(particles.data(), particles.size(), pos2Dvel2Dcol3Dmass);
            
        }
    }

private:
    std::tuple<float, float> checkError()
    {
        float noAccumulator = 0.0f;

        nBodySolverNaive.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);

        nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
        float error1 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error1 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error1 /= particles.size();

        nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.15f);
        float error2 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error2 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error2 /= particles.size();

        return std::make_tuple(error1 - error2, error1 / error2);
    }

};