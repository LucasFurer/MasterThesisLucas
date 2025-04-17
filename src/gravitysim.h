#pragma once
#include <vector>
#include "particles/particle3D.h"
#include "buffer.h"
#include "nbodysolvers/nBodySolverNaive.h"
#include "nbodysolvers/nBodySolverBarnesHut.h"
#include "nbodysolvers/nBodySolverBarnesHutReverse.h"
#include "nbodysolvers/nBodySolverMultiPole.h"
#include "nbodysolvers/nBodySolverBarnesHutReverseMultiPole.h"
#include "nbodysolvers/nBodySolverFMM.h"

class GravitySim
{
public:
	std::vector<Particle2D> particles;
	std::vector<glm::vec2> accelerations;
	std::vector<glm::vec2> accelerationsErrorTest;
	Buffer* particlesBuffer;

    float stepSize = 0.1f;
    float timeStepsPerSec = 300000000.0f;
    float lastTimeUpdated = 0.0f;

    NBodySolverNaive<Particle2D> nBodySolverNaive;
    NBodySolverBarnesHut<Particle2D> nBodySolverBarnesHut;
    NBodySolverBarnesHutReverse<Particle2D> nBodySolverBarnesHutReverse;
    NBodySolverMultiPole<Particle2D> nBodySolverMultiPole;
    NBodySolverBarnesHutReverseMultiPole<Particle2D> nBodySolverBarnesHutReverseMultiPole;
    NBodySolverFMM<Particle2D> nBodySolverFMM;

	GravitySim(int particleAmount)
	{
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        nBodySolverNaive = NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
        nBodySolverBarnesHut = NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal);
        nBodySolverBarnesHutReverse = NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal);
        nBodySolverMultiPole = NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal);
        nBodySolverBarnesHutReverseMultiPole = NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal);
        nBodySolverFMM = NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal);

        srand(1952731);
        float sizeParam = 200.0f;
        float velParam = 1.0f;
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


            //float initVelMag = 15.0f;
            float randXV = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randYV = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;

            while (powf(randXV, 2.0f) + powf(randYV, 2.0f) > 1.0f)
            {
                randXV = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randYV = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }

            glm::vec2 vel = glm::vec2(
                velParam * randXV,
                velParam * randYV
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

            //checkError();

            float noAccumulator = 0.0f;
            //nBodySolverNaive.solveNbody(&noAccumulator, &accelerations, &particles);
            //nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerations, &particles, 10, 0.8f); // keep theta between 0.0 (off) and 1.0 (can be higher) 0.3 gives no artifacts
            //nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
            //nBodySolverBarnesHutReverseMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
            nBodySolverFMM.solveNbody(&noAccumulator, &accelerations, &particles, 10, 0.8f);

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
    void checkError()
    {
        float noAccumulator = 0.0f;

        nBodySolverNaive.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);

        //nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
        nBodySolverBarnesHutReverse.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
        float error1 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error1 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error1 /= particles.size();

        //nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
        nBodySolverBarnesHutReverseMultiPole.solveNbody(&noAccumulator, &accelerations, &particles, 10, 1.0f);
        //nBodySolverFMM.solveNbody(&noAccumulator, &accelerations, &particles, 10, 0.6f);
        float error2 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error2 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error2 /= particles.size();



        std::cout << "difference in error: " << error1 - error2 << " ,greater than 0.0 is good" << std::endl;
        std::cout << "ratio of error:      " << error1 / error2 << " ,greater than 1.0 is good" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
    }

};