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

class NBodyScenarios
{
public:
    std::vector<Particle2D> particles;
    std::vector<glm::vec2> accelerations;
    std::vector<glm::vec2> accelerationsErrorTest;
    Buffer* particlesBuffer;

    float stepSize = 0.1f;
    float timeStepsPerSec = 30.0f;
    float lastTimeUpdated = 0.0f;

    NBodySolverNaive<Particle2D> nBodySolverNaive = NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
    NBodySolverBarnesHut<Particle2D> nBodySolverBarnesHut = NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal);
    NBodySolverBarnesHutReverse<Particle2D> nBodySolverBarnesHutReverse = NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal);
    NBodySolverMultiPole<Particle2D> nBodySolverMultiPole = NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal);
    NBodySolverBarnesHutReverseMultiPole<Particle2D> nBodySolverBarnesHutReverseMultiPole = NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal);
    //NBodySolverFMM<Particle2D> nBodySolverFMM = NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernalNaive, &GRAVITYFMMParticleParticleKernal);
    NBodySolverFMM<Particle2D> nBodySolverFMM = NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernal, &GRAVITYFMMParticleNodeKernal, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal);

    NBodyScenarios()
    {

    }

    ~NBodyScenarios()
    {
        // delete embeddedBuffer?
    }

    void cleanup()
    {
        particlesBuffer->cleanup();
    }


    void ErrorTimestep()
    {
        generatePoints(100);

        // set graph size
        int errorMeasurementAmount = 1000;
        std::vector<float> errorBH(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHMultipole(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHReverse(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHReverseMultipole(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMM(errorMeasurementAmount, 0.0f);

        std::vector<int> timeBH(errorMeasurementAmount, 0);
        std::vector<int> timeBHMultipole(errorMeasurementAmount, 0);
        std::vector<int> timeBHReverse(errorMeasurementAmount, 0);
        std::vector<int> timeBHReverseMultipole(errorMeasurementAmount, 0);
        std::vector<int> timeFMM(errorMeasurementAmount, 0);


        // find error at every time step
        float noAccumulator = 0.0f;
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            //lastTimeUpdated = glfwGetTime();
            // correct solution up to machine precision
            nBodySolverNaive.solveNbody(&noAccumulator, &accelerations, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, 1.0f);
            timeBH[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBH[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBH[t] /= particles.size();
            
            nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, 1.0f);
            timeBHMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHMultipole[t] /= particles.size();

            nBodySolverBarnesHutReverse.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, 1.0f);
            timeBHReverse[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverse[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverse[t] /= particles.size();

            nBodySolverBarnesHutReverseMultiPole.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, 1.0f);
            timeBHReverseMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverseMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverseMultipole[t] /= particles.size();

            nBodySolverFMM.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, 1.0f);
            timeFMM[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBufferNew(particles.data(), particles.size(), pos2Dvel2Dcol3Dmass);
        }

        // write results to csv files
        std::ofstream file1("graphCSV/lineErrorTimestepBH.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file1 << timeBH[i] << "," << errorBH[i] << "\n";

        std::ofstream file2("graphCSV/lineErrorTimestepBHMultipole.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file2 << timeBHMultipole[i] << "," << errorBHMultipole[i] << "\n";

        std::ofstream file3("graphCSV/lineErrorTimestepBHReverse.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file3 << timeBHReverse[i] << "," << errorBHReverse[i] << "\n";

        std::ofstream file4("graphCSV/lineErrorTimestepBHReverseMultipole.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file4 << timeBHReverseMultipole[i] << "," << errorBHReverseMultipole[i] << "\n";

        std::ofstream file5("graphCSV/lineErrorTimestepFMM.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file5 << timeFMM[i] << "," << errorFMM[i] << "\n";
    }

    /*
    void calculationtimeTheta()
    {
        generatePoints(100);

        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;

        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaBH(thetaDiversityAmount, 0);
        std::vector<float> thetaBHMultipole(thetaDiversityAmount, 0);
        std::vector<float> thetaBHReverse(thetaDiversityAmount, 0);
        std::vector<float> thetaBHReverseMultipole(thetaDiversityAmount, 0);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0);


        // find error at every time step
        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            //lastTimeUpdated = glfwGetTime();
            // correct solution up to machine precision
            nBodySolverNaive.solveNbody(&noAccumulator, &accelerations, &particles);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolverBarnesHut.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, chosenTheta);
            thetaBH[t] = chosenTheta;
            for (int i = 0; i < particles.size(); i++)
                errorBH[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBH[t] /= particles.size();

            nBodySolverMultiPole.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, chosenTheta);
            thetaBHMultipole[t] = chosenTheta;
            for (int i = 0; i < particles.size(); i++)
                errorBHMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHMultipole[t] /= particles.size();

            nBodySolverBarnesHutReverse.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, chosenTheta);
            thetaBHReverse[t] = chosenTheta;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverse[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverse[t] /= particles.size();

            nBodySolverBarnesHutReverseMultiPole.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, chosenTheta);
            thetaBHReverseMultipole[t] = chosenTheta;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverseMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverseMultipole[t] /= particles.size();

            nBodySolverFMM.solveNbody(&noAccumulator, &accelerationsErrorTest, &particles, 10, chosenTheta);
            thetaFMM[t] = chosenTheta;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBufferNew(particles.data(), particles.size(), pos2Dvel2Dcol3Dmass);
        }

        // write results to csv files
        std::ofstream file1("graphCSV/lineErrorTimestepBH.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file1 << timeBH[i] << "," << errorBH[i] << "\n";

        std::ofstream file2("graphCSV/lineErrorTimestepBHMultipole.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file2 << timeBHMultipole[i] << "," << errorBHMultipole[i] << "\n";

        std::ofstream file3("graphCSV/lineErrorTimestepBHReverse.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file3 << timeBHReverse[i] << "," << errorBHReverse[i] << "\n";

        std::ofstream file4("graphCSV/lineErrorTimestepBHReverseMultipole.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file4 << timeBHReverseMultipole[i] << "," << errorBHReverseMultipole[i] << "\n";

        std::ofstream file5("graphCSV/lineErrorTimestepFMM.csv");
        for (size_t i = 0; i < errorMeasurementAmount; ++i)
            file5 << timeFMM[i] << "," << errorFMM[i] << "\n";
    }
    */

private:
    void generatePoints(int particleAmount)
    {
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        srand(1952731);
        float sizeParam = sqrt(particleAmount)/0.16f;
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
};