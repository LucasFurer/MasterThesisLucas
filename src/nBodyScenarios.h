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

    std::map<std::string, NBodySolver<Particle2D>*> nBodySolvers;

    NBodyScenarios()
    {
        nBodySolvers["naive"] = new NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
        nBodySolvers["BH"] = new NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHR"] = new NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHMP"] = new NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["FMM"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernal, &GRAVITYFMMParticleNodeKernal, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);
        nBodySolvers["FMMnaive"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernalNaive, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);
    }

    ~NBodyScenarios()
    {
        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }
    }

    void cleanup()
    {
        particlesBuffer->cleanup();

        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolver : nBodySolvers)
        {
            nBodySolver.second->boxBuffer->cleanup();
        }
    }

    void testNodeNode()
    {
        std::vector<Particle2D> particles(20);
        std::vector<glm::vec2> trueAccelerations(20);
        std::vector<glm::vec2> stupidNodeNodeAccelerations(20);
        std::vector<glm::vec2> smartNodeNodeAccelerations(20);

        {
            particles[0] = Particle2D(glm::vec2(0.12, -0.85), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[1] = Particle2D(glm::vec2(0.43, 1.22), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[2] = Particle2D(glm::vec2(-1.18, -0.39), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[3] = Particle2D(glm::vec2(0.57, -0.01), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[4] = Particle2D(glm::vec2(-0.62, 0.87), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[5] = Particle2D(glm::vec2(-0.95, 0.11), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[6] = Particle2D(glm::vec2(1.30, -1.46), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[7] = Particle2D(glm::vec2(0.05, 0.60), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[8] = Particle2D(glm::vec2(-0.47, -0.23), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[9] = Particle2D(glm::vec2(0.88, 0.31), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            for (int i = 0; i < 10; i++) { particles[i].position.x += 1.0f; }

            particles[10] = Particle2D(glm::vec2(-0.34, 0.92), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[11] = Particle2D(glm::vec2(1.12, -0.77), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[12] = Particle2D(glm::vec2(0.26, 0.04), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[13] = Particle2D(glm::vec2(-0.61, 1.43), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[14] = Particle2D(glm::vec2(0.90, -0.16), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[15] = Particle2D(glm::vec2(-1.25, -1.01), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[16] = Particle2D(glm::vec2(0.37, 0.68), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[17] = Particle2D(glm::vec2(-0.08, -0.54), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[18] = Particle2D(glm::vec2(0.13, 1.05), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            particles[19] = Particle2D(glm::vec2(-0.73, 0.22), glm::vec2(0.0f), glm::vec3(1.0f), 1.0f);
            for (int i = 10; i < 20; i++) { particles[i].position.x -= 1.0f; }
        }

        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                float softening = 0.1f;
                glm::vec2 diff = particles[i].position - particles[j].position;
                float distance = glm::length(diff);

                float oneOverDistance = 1.0f / (distance + softening);
                trueAccelerations[i] += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
            }
        }

        glm::vec2 centreOfMass1 = glm::vec2(0.0f);
        float totalMass1 = 0.0f;
        glm::vec2 dipole1 = glm::vec2(0.0f);
        Fastor::Tensor<float, 2, 2> quadrupole1{};
        {
            for (int i = 0; i < 10; i++)
            {
                centreOfMass1 += particles[i].position;
                totalMass1 += particles[i].mass;
            }
            centreOfMass1 /= totalMass1;
            for (int i = 0; i < 10; i++)
            {
                glm::vec2 relativeCoord = particles[i].position - centreOfMass1;
                dipole1 += relativeCoord;

                Fastor::Tensor<float, 2, 2> outer_product;
                outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                quadrupole1 += outer_product;
            }
        }

        glm::vec2 centreOfMass2 = glm::vec2(0.0f);
        float totalMass2 = 0.0f;
        glm::vec2 dipole2 = glm::vec2(0.0f);
        Fastor::Tensor<float, 2, 2> quadrupole2{};
        {
            for (int i = 10; i < 20; i++)
            {
                centreOfMass2 += particles[i].position;
                totalMass2 += particles[i].mass;
            }
            centreOfMass2 /= totalMass2;
            for (int i = 10; i < 20; i++)
            {
                glm::vec2 relativeCoord = particles[i].position - centreOfMass2;
                dipole2 += relativeCoord;

                Fastor::Tensor<float, 2, 2> outer_product;
                outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                quadrupole2 += outer_product;
            }
        }

        std::cout << "centre of mass 1: " << glm::to_string(centreOfMass1) << std::endl;
        std::cout << "centre of mass 2: " << glm::to_string(centreOfMass2) << std::endl;
    }

    void errorTimestep()
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
            nBodySolvers["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolvers["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBH[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBH[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBH[t] /= particles.size();
            
            nBodySolvers["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHMultipole[t] /= particles.size();

            nBodySolvers["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHReverse[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverse[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverse[t] /= particles.size();

            nBodySolvers["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHReverseMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverseMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverseMultipole[t] /= particles.size();

            nBodySolvers["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);//, 10, 1.0f
            timeFMM[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);
        }

        // write results to csv files
        writeToFile(timeBH, errorBH, "graphCSV/scenario1lineErrorTimestepBH.csv");
        writeToFile(timeBHMultipole, errorBHMultipole, "graphCSV/scenario1lineErrorTimestepBHMultipole.csv");
        writeToFile(timeBHReverse, errorBHReverse, "graphCSV/scenario1lineErrorTimestepBHReverse.csv");
        writeToFile(timeBHReverseMultipole, errorBHReverseMultipole, "graphCSV/scenario1lineErrorTimestepBHReverseMultipole.csv");
        writeToFile(timeFMM, errorFMM, "graphCSV/scenario1lineErrorTimestepFMM.csv");
    }

    void errorTimestepFMM()
    {
        generatePoints(1000);

        // set graph size
        int errorMeasurementAmount = 500;
        std::vector<float> errorFMM(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMMnaive(errorMeasurementAmount, 0.0f);

        std::vector<int> timeFMM(errorMeasurementAmount, 0);
        std::vector<int> timeFMMnaive(errorMeasurementAmount, 0);


        // find error at every time step
        float noAccumulator = 0.0f;
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            //lastTimeUpdated = glfwGetTime();
            // correct solution up to machine precision
            nBodySolvers["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
             nBodySolvers["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeFMM[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            nBodySolvers["FMMnaive"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeFMMnaive[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMMnaive[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMMnaive[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);
        }

        // write results to csv files
        writeToFile(timeFMM, errorFMM, "graphCSV/scenario2lineErrorTimestepFMMcompare_FMM.csv");
        writeToFile(timeFMMnaive, errorFMMnaive, "graphCSV/scenario2lineErrorTimestepFMMcompare_FMMnaive.csv");
    }

    void calculationtimeTheta()
    {
        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;

        int averageOverAmount = 100;

        std::vector<float> calculationtimeNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);


        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePoints(1000);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMultipole[t] = chosenTheta;
            thetaBHReverse[t] = chosenTheta;
            thetaBHReverseMultipole[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolvers["BH"]->theta = chosenTheta;
            nBodySolvers["BHMP"]->theta = chosenTheta;
            nBodySolvers["BHR"]->theta = chosenTheta;
            nBodySolvers["BHRMP"]->theta = chosenTheta;
            nBodySolvers["FMM"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for(int i = 0; i < averageOverAmount; i++)
            { 
                timeBefore = glfwGetTime();
                nBodySolvers["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);
                calculationtimeNaive[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverse[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverseMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;

                particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);
            }

            calculationtimeNaive[t] /= averageOverAmount;
            calculationtimeBH[t] /= averageOverAmount;
            calculationtimeBHMultipole[t] /= averageOverAmount;
            calculationtimeBHReverse[t] /= averageOverAmount;
            calculationtimeBHReverseMultipole[t] /= averageOverAmount;
            calculationtimeFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        writeToFile(thetaNaive, calculationtimeNaive, ("graphCSV/scenario3calculationtimeThetaNaive.csv"));
        writeToFile(thetaBH, calculationtimeBH, ("graphCSV/scenario3calculationtimeThetaBH.csv"));
        writeToFile(thetaBHMultipole, calculationtimeBHMultipole, ("graphCSV/scenario3calculationtimeThetaBHMP.csv"));
        writeToFile(thetaBHReverse, calculationtimeBHReverse, ("graphCSV/scenario3calculationtimeThetaBHR.csv"));
        writeToFile(thetaBHReverseMultipole, calculationtimeBHReverseMultipole, ("graphCSV/scenario3calculationtimeThetaBHRMP.csv"));
        writeToFile(thetaFMM, calculationtimeFMM, ("graphCSV/scenario3calculationtimeThetaFMM.csv"));
    }

    void errorTheta()
    {
        /*
        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;

        int averageOverAmount = 100;

        std::vector<float> calculationtimeNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);


        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePoints(1000);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMultipole[t] = chosenTheta;
            thetaBHReverse[t] = chosenTheta;
            thetaBHReverseMultipole[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolvers["BH"]->theta = chosenTheta;
            nBodySolvers["BHMP"]->theta = chosenTheta;
            nBodySolvers["BHR"]->theta = chosenTheta;
            nBodySolvers["BHRMP"]->theta = chosenTheta;
            nBodySolvers["FMM"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for (int i = 0; i < averageOverAmount; i++)
            {
                timeBefore = glfwGetTime();
                nBodySolvers["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);
                calculationtimeNaive[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverse[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverseMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolvers["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;

                particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);
            }

            calculationtimeNaive[t] /= averageOverAmount;
            calculationtimeBH[t] /= averageOverAmount;
            calculationtimeBHMultipole[t] /= averageOverAmount;
            calculationtimeBHReverse[t] /= averageOverAmount;
            calculationtimeBHReverseMultipole[t] /= averageOverAmount;
            calculationtimeFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        writeToFile(thetaNaive, calculationtimeNaive, ("graphCSV/scenario3calculationtimeThetaNaive.csv"));
        writeToFile(thetaBH, calculationtimeBH, ("graphCSV/scenario3calculationtimeThetaBH.csv"));
        writeToFile(thetaBHMultipole, calculationtimeBHMultipole, ("graphCSV/scenario3calculationtimeThetaBHMP.csv"));
        writeToFile(thetaBHReverse, calculationtimeBHReverse, ("graphCSV/scenario3calculationtimeThetaBHR.csv"));
        writeToFile(thetaBHReverseMultipole, calculationtimeBHReverseMultipole, ("graphCSV/scenario3calculationtimeThetaBHRMP.csv"));
        writeToFile(thetaFMM, calculationtimeFMM, ("graphCSV/scenario3calculationtimeThetaFMM.csv"));
        */
    }

private:
    template <typename T, typename I>
    void writeToFile(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, std::string filename)
    {
        std::ofstream file(filename);
        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "\n";
    }

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

        particlesBuffer = new Buffer(particles, pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
    }
};