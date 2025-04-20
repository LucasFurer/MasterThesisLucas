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
#include "nbodysolvers/nBodySolver.h"

class GravitySim
{
public:
	std::vector<Particle2D> particles;
	std::vector<glm::vec2> accelerations;
	std::vector<glm::vec2> accelerationsErrorTest;
	Buffer* particlesBuffer;
    Buffer* forceBuffer;

    float stepSize = 0.1f;
    float timeStepsPerSec = 3.0f;
    float lastTimeUpdated = 0.0f;

    std::vector<NBodySolver<Particle2D>*> nBodySolvers;
    int nBodySelect = 0;

	GravitySim(int particleAmount)
	{
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        nBodySolvers.push_back(new NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal));
        nBodySolvers.push_back(new NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal,                                                   10, 1.0f));
        nBodySolvers.push_back(new NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal,                              10, 1.0f));
        nBodySolvers.push_back(new NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal,                                                   10, 1.0f));
        nBodySolvers.push_back(new NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal,   10, 1.0f));
        nBodySolvers.push_back(new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal, 10, 1.0f));

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

        particlesBuffer = new Buffer(particles, pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
        
        std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(particles, accelerations);
        forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
	}

    ~GravitySim()
    {
        delete particlesBuffer;
        delete forceBuffer;

        for (NBodySolver<Particle2D>* nBodySolverPointer : nBodySolvers) 
        {
            delete nBodySolverPointer;
        }
    }

    void cleanup()
    {
        particlesBuffer->cleanup();
        forceBuffer->cleanup();

        for (NBodySolver<Particle2D>* nBodySolver : nBodySolvers)
        {
            nBodySolver->boxBuffer->cleanup();
        }
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
            nBodySolvers[nBodySelect]->solveNbody(&noAccumulator, &accelerations, &particles);

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

            particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

            std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(particles, accelerations);
            forceBuffer->updateBuffer(forceLines, pos2DCol3D);
        }
    }

private:
    void checkError()
    {
        float noAccumulator = 0.0f;

        nBodySolvers[0]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);

        nBodySolvers[1]->solveNbody(&noAccumulator, &accelerations, &particles);
        float error1 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error1 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error1 /= particles.size();

        //nBodySolvers[3]->solveNbody(&noAccumulator, &accelerations, &particles);
        nBodySolvers[2]->solveNbody(&noAccumulator, &accelerations, &particles);
        //nBodySolvers[5]->.solveNbody(&noAccumulator, &accelerations, &particles);
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