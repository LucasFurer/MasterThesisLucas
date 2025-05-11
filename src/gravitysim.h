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
    float forceSize = 1.0f;

    float stepSize = 0.1f;
    float timeStepsPerSec = 99999999.0f;
    float lastTimeUpdated = 0.0f;

    std::map<std::string, NBodySolver<Particle2D>*> nBodySolvers;
    std::string nBodySelect = "naive";

    float totalError1 = 0.0f;
    float totalError2 = 0.0f;
    float maxError1 = 0.0f;
    float maxError2 = 0.0f;
    int globalTimeStep = 0;

	GravitySim(int particleAmount)
	{
        generatePoints(particleAmount, 1952731);
        //generatePointsCustom1();

        nBodySolvers["naive"] = new NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
        nBodySolvers["BH"] = new NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BH"]->updateTree(&particles);
        nBodySolvers["BHR"] = new NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHR"]->updateTree(&particles);
        nBodySolvers["BHMP"] = new NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHMP"]->updateTree(&particles);
        nBodySolvers["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHRMP"]->updateTree(&particles);
        nBodySolvers["FMM"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernal, &GRAVITYFMMParticleNodeKernal, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);
        nBodySolvers["FMM"]->updateTree(&particles);
        nBodySolvers["FMMnaive"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernalNaive, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);
        nBodySolvers["FMMnaive"]->updateTree(&particles);


        particlesBuffer = new Buffer(particles, pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
        
        std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(particles, accelerations, forceSize);
        forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
	}

    ~GravitySim()
    {
        delete particlesBuffer;
        delete forceBuffer;

        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }
    }

    void cleanup()
    {
        particlesBuffer->cleanup();
        forceBuffer->cleanup();

        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolver : nBodySolvers)
        {
            nBodySolver.second->boxBuffer->cleanup();
        }
    }


    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            //checkError();

            float noAccumulator = 0.0f;
            nBodySolvers[nBodySelect]->theta = 2.5f;
            nBodySolvers[nBodySelect]->solveNbody(&noAccumulator, &accelerations, &particles);
            globalTimeStep++;


            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];
            
            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

            nBodySolvers[nBodySelect]->updateTree(&particles);

            std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(particles, accelerations, forceSize);
            forceBuffer->updateBuffer(forceLines, pos2DCol3D);
        }
    }

private:
    void checkError()
    {
        float noAccumulator = 0.0f;

        nBodySolvers["naive"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);

        nBodySolvers["BH"]->theta = 1.0f;
        nBodySolvers["BH"]->solveNbody(&noAccumulator, &accelerations, &particles);
        float error1 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error1 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error1 /= particles.size();
        totalError1 += error1;
        if (error1 > maxError1) { maxError1 = error1; }

        nBodySolvers["FMM"]->theta = 1.3f;
        nBodySolvers["FMM"]->solveNbody(&noAccumulator, &accelerations, &particles);
        float error2 = 0.0f;
        for (int i = 0; i < particles.size(); i++)
        {
            error2 += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
        }
        error2 /= particles.size();
        totalError2 += error2;
        if (error2 > maxError2) { maxError2 = error2; }


        std::cout << "difference in error:         " << error1 - error2 << " ,greater than 0.0 is good" << std::endl;
        std::cout << "average error difference:    " << (totalError1 / globalTimeStep) - (totalError2 / globalTimeStep) << " ,greater than 0.0 is good" << std::endl;

        std::cout << "max error1:                  " << maxError1 << std::endl;
        std::cout << "max error2:                  " << maxError2 << std::endl;

        std::cout << "ratio of error:              " << error1 / error2 << " ,greater than 1.0 is good" << std::endl;
        std::cout << "ratio of average error:      " << (totalError1 / globalTimeStep) / (totalError2 / globalTimeStep) << " ,greater than 1.0 is good" << std::endl;

        std::cout << "global time step: " << globalTimeStep << std::endl;

        std::cout << "------------------------------------------------------" << std::endl;
    }

    void generatePoints(int particleAmount, int seed)
    {
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        srand(seed);
        float sizeParam = sqrt(particleAmount) / 0.16f;
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
    }

    void generatePointsCustom1()
    {
        int particleAmount = 40;
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        std::vector<glm::vec2> positions;
        float distance = 5.0f;

        positions.push_back(glm::vec2(0.4967f, -0.1383f));
        positions.push_back(glm::vec2(0.6477f, 1.5230f));
        positions.push_back(glm::vec2(-0.2342f, -0.2341f));
        positions.push_back(glm::vec2(1.5792f, 0.7674f));
        positions.push_back(glm::vec2(-0.4695f, 0.5426f));
        positions.push_back(glm::vec2(-0.4634f, -0.4657f));
        positions.push_back(glm::vec2(0.2420f, -1.9133f));
        positions.push_back(glm::vec2(-1.7249f, -0.5623f));
        positions.push_back(glm::vec2(-1.0128f, 0.3142f));
        positions.push_back(glm::vec2(-0.9080f, -1.4123f));
        for (int i = 0; i < 10; i++) { positions[i] += glm::vec2(distance, distance); }

        positions.push_back(glm::vec2(1.4656f, -0.2258f));
        positions.push_back(glm::vec2(0.0675f, -1.4247f));
        positions.push_back(glm::vec2(-0.5444f, 0.1109f));
        positions.push_back(glm::vec2(-1.1510f, 0.3757f));
        positions.push_back(glm::vec2(-0.6006f, -0.2917f));
        positions.push_back(glm::vec2(-0.6017f, 1.8523f));
        positions.push_back(glm::vec2(-0.0135f, -1.0577f));
        positions.push_back(glm::vec2(0.8225f, -1.2208f));
        positions.push_back(glm::vec2(0.2089f, -1.9597f));
        positions.push_back(glm::vec2(-1.3282f, 0.1969f));
        for (int i = 10; i < 20; i++) { positions[i] += glm::vec2(distance, -distance); }

        positions.push_back(glm::vec2(0.7385f, 0.1714f));
        positions.push_back(glm::vec2(-0.1156f, -0.3011f));
        positions.push_back(glm::vec2(-1.4785f, -0.7198f));
        positions.push_back(glm::vec2(-0.4606f, 1.0571f));
        positions.push_back(glm::vec2(0.3436f, -1.7630f));
        positions.push_back(glm::vec2(0.3241f, -0.3851f));
        positions.push_back(glm::vec2(-0.6769f, 0.6117f));
        positions.push_back(glm::vec2(1.0310f, 0.9313f));
        positions.push_back(glm::vec2(-0.8392f, -0.3092f));
        positions.push_back(glm::vec2(0.3313f, 0.9755f));
        for (int i = 20; i < 30; i++) { positions[i] += glm::vec2(-distance, distance); }

        positions.push_back(glm::vec2(-0.4792f, -0.1857f));
        positions.push_back(glm::vec2(-1.1063f, -1.1962f));
        positions.push_back(glm::vec2(0.8125f, 1.3562f));
        positions.push_back(glm::vec2(-0.0720f, 1.0035f));
        positions.push_back(glm::vec2(0.3616f, -0.6451f));
        positions.push_back(glm::vec2(0.3614f, 1.5380f));
        positions.push_back(glm::vec2(-0.0358f, 1.5646f));
        positions.push_back(glm::vec2(-2.6197f, 0.8219f));
        positions.push_back(glm::vec2(0.0870f, -0.2990f));
        positions.push_back(glm::vec2(0.0918f, -1.9876f));
        for (int i = 30; i < 40; i++) { positions[i] += glm::vec2(-distance, -distance); }

        float velParam = 0.0f;
        for (int i = 0; i < particleAmount; i++)
        {
            glm::vec2 pos = positions[i];


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
    }

};