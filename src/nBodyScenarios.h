#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
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
    //Buffer* particlesBuffer;

    float stepSize = 0.1f;
    float timeStepsPerSec = 30.0f;
    float lastTimeUpdated = 0.0f;

    std::map<std::string, NBodySolver<Particle2D>*> nBodySolversGRAVITY;





    std::vector<EmbeddedPoint> embeddedPoints;
    std::vector<EmbeddedPoint> embeddedPointsPrev;
    std::vector<EmbeddedPoint> embeddedPointsPrevPrev;

    std::vector<glm::vec2> embeddedDerivative;
    std::vector<glm::vec2> embeddedDerivativeErrorTest;

    std::vector<glm::vec2> attractForce;

    std::vector<glm::vec2> repulsForce;
    std::vector<glm::vec2> repulsForceErrorTest;
    std::vector<glm::vec2> repulsForceNotNorm;
    std::vector<glm::vec2> repulsForceErrorTestNotNorm;



    std::map<std::string, NBodySolver<EmbeddedPoint>*> nBodySolversTSNE;
    std::string nBodySelect = "naive";

    float learnRate;
    float accelerationRate;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;





    NBodyScenarios()
    {
        nBodySolversGRAVITY["naive"] = new NBodySolverNaive<Particle2D>(&GRAVITYnaiveKernal);
        nBodySolversGRAVITY["BH"] = new NBodySolverBarnesHut<Particle2D>(&GRAVITYbarnesHutParticleNodeKernal, &GRAVITYbarnesHutParticleParticleKernal, 10, 1.0f);
        nBodySolversGRAVITY["BHR"] = new NBodySolverBarnesHutReverse<Particle2D>(&GRAVITYbarnesHutReverseParticleNodeKernal, &GRAVITYbarnesHutReverseParticleParticleKernal, 10, 1.0f);
        nBodySolversGRAVITY["BHMP"] = new NBodySolverMultiPole<Particle2D>(&GRAVITYmultiPoleParticleNodeKernal, &GRAVITYmultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolversGRAVITY["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<Particle2D>(&GRAVITYbarnesHutReverseMultiPoleParticleNodeKernal, &GRAVITYbarnesHutReverseMultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolversGRAVITY["FMM"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernal, &GRAVITYFMMParticleNodeKernal, &GRAVITYFMMNodeParticleKernal, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);
        nBodySolversGRAVITY["FMMnaive"] = new NBodySolverFMM<Particle2D>(&GRAVITYFMMNodeNodeKernalNaive, &GRAVITYFMMParticleNodeKernalNaive, &GRAVITYFMMNodeParticleKernalNaive, &GRAVITYFMMParticleParticleKernal, 10, 1.0f);

        nBodySolversTSNE["naive"] = new NBodySolverNaive<EmbeddedPoint>(&TSNEnaiveKernal);
        nBodySolversTSNE["BH"] = new NBodySolverBarnesHut<EmbeddedPoint>(&TSNEbarnesHutParticleNodeKernal, &TSNEbarnesHutParticleParticleKernal, 10, 1.0f);
        nBodySolversTSNE["BHR"] = new NBodySolverBarnesHutReverse<EmbeddedPoint>(&TSNEbarnesHutReverseParticleNodeKernal, &TSNEbarnesHutReverseParticleParticleKernal, 10, 1.0f);
        nBodySolversTSNE["BHMP"] = new NBodySolverMultiPole<EmbeddedPoint>(&TSNEmultiPoleParticleNodeKernal, &TSNEmultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolversTSNE["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<EmbeddedPoint>(&TSNEbarnesHutReverseMultiPoleParticleNodeKernal, &TSNEbarnesHutReverseMultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolversTSNE["FMM"] = new NBodySolverFMM<EmbeddedPoint>(&TSNEFMMNodeNodeKernal, &TSNEFMMParticleNodeKernal, &TSNEFMMNodeParticleKernal, &TSNEFMMParticleParticleKernal, 10, 1.0f);
        nBodySolversTSNE["FMMnaive"] = new NBodySolverFMM<EmbeddedPoint>(&TSNEFMMNodeNodeKernalNaive, &TSNEFMMParticleNodeKernalNaive, &TSNEFMMNodeParticleKernalNaive, &TSNEFMMParticleParticleKernal, 10, 1.0f);
    }

    ~NBodyScenarios()
    {
        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolverPointer : nBodySolversGRAVITY)
        {
            delete nBodySolverPointer.second;
        }

        for (std::pair<const std::string, NBodySolver<EmbeddedPoint>*> nBodySolverPointer : nBodySolversTSNE)
        {
            delete nBodySolverPointer.second;
        }
    }

    void cleanup()
    {
        //particlesBuffer->cleanup();

        for (std::pair<const std::string, NBodySolver<Particle2D>*> nBodySolver : nBodySolversGRAVITY)
        {
            nBodySolver.second->boxBuffer->cleanup();
        }

        for (std::pair<const std::string, NBodySolver<EmbeddedPoint>*> nBodySolver : nBodySolversTSNE)
        {
            nBodySolver.second->boxBuffer->cleanup();
        }
    }

    void testNodeNode()
    {
        // create room for particles
        std::vector<Particle2D> particles(20);
        std::vector<glm::vec2> trueAccelerations(20);
        std::vector<glm::vec2> stupidNodeNodeAccelerations(20);
        std::vector<glm::vec2> smartNodeNodeAccelerations(20);

        // initialize particles
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
            for (int i = 0; i < 10; i++) { particles[i].position.x += 5.0f; }

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
            for (int i = 10; i < 20; i++) { particles[i].position.x -= 5.0f; }
        }

        // calculate true acceleration
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

        // poles of 1
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

        // poles of 1
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

        // naive interaction of cluster 1
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                float softening = 0.1f;
                glm::vec2 diff = particles[i].position - particles[j].position;
                float distance = glm::length(diff);

                float oneOverDistance = 1.0f / (distance + softening);
                stupidNodeNodeAccelerations[i] += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
                smartNodeNodeAccelerations[i] += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
            }
        }

        // naive interaction of cluster 2
        for (int i = 10; i < 20; i++)
        {
            for (int j = 10; j < 20; j++)
            {
                float softening = 0.1f;
                glm::vec2 diff = particles[i].position - particles[j].position;
                float distance = glm::length(diff);

                float oneOverDistance = 1.0f / (distance + softening);
                stupidNodeNodeAccelerations[i] += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
                smartNodeNodeAccelerations[i] += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
            }
        }

        // stupid node node
        {
            float softening = 0.1f;
            glm::vec2 diff = centreOfMass1 - centreOfMass2;
            float distance = glm::length(diff);

            float oneOverDistance = (1.0f / (distance + softening));
            for (int i = 0; i < 10; i++)
            {
                stupidNodeNodeAccelerations[i] += -totalMass2 * oneOverDistance * oneOverDistance * oneOverDistance * diff;
            }

            //----------------------------------------------------------------------------------------------------------------

            softening = 0.1f;
            diff = centreOfMass2 - centreOfMass1;
            distance = glm::length(diff);

            oneOverDistance = (1.0f / (distance + softening));
            for (int i = 10; i < 20; i++)
            {
                stupidNodeNodeAccelerations[i] += -totalMass1 * oneOverDistance * oneOverDistance * oneOverDistance * diff;
            }
        }

        // smart node node
        smartNodeNode(particles, smartNodeNodeAccelerations, centreOfMass1, totalMass1, dipole1, quadrupole1, centreOfMass2, totalMass2, dipole2, quadrupole2, true);
        smartNodeNode(particles, smartNodeNodeAccelerations, centreOfMass2, totalMass2, dipole2, quadrupole2, centreOfMass1, totalMass1, dipole1, quadrupole1, false);


        // error of all
        float errorStupid = 0.0f;
        float errorSmart = 0.0f;
        {
            for (int i = 0; i < 20; i++)
            {
                errorStupid += powf(glm::length(trueAccelerations[i] - stupidNodeNodeAccelerations[i]), 2.0f);
            }
            errorStupid /= 20;

            for (int i = 0; i < 20; i++)
            {
                errorSmart += powf(glm::length(trueAccelerations[i] - smartNodeNodeAccelerations[i]), 2.0f);
            }
            errorSmart /= 20;
        }

        std::cout << "stupid error: " << errorStupid << std::endl;
        std::cout << "smart error: " << errorSmart << std::endl;
    }

    void errorTimestep()
    {
        int errorMeasurementAmount = 100; // how many iterations to run the simulation
        int particleCount = 100; // use this many particles

        generatePoints(particleCount);
        nBodySolversGRAVITY["BH"]->updateTree(&particles);
        nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
        nBodySolversGRAVITY["BHR"]->updateTree(&particles);
        nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
        nBodySolversGRAVITY["FMM"]->updateTree(&particles);


        // set graph size
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
            nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBH[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBH[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBH[t] /= particles.size();
            
            nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHMultipole[t] /= particles.size();

            nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHReverse[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverse[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverse[t] /= particles.size();

            nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeBHReverseMultipole[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorBHReverseMultipole[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorBHReverseMultipole[t] /= particles.size();

            nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);//, 10, 1.0f
            timeFMM[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            //particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

            nBodySolversGRAVITY["BH"]->updateTree(&particles);
            nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
            nBodySolversGRAVITY["BHR"]->updateTree(&particles);
            nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
            nBodySolversGRAVITY["FMM"]->updateTree(&particles);
        }

        // write results to csv files
        // writeToFile(timeBH, errorBH, "graphCSV/scenario1lineErrorTimestepBH.csv");
        // writeToFile(timeBHMultipole, errorBHMultipole, "graphCSV/scenario1lineErrorTimestepBHMultipole.csv");
        // writeToFile(timeBHReverse, errorBHReverse, "graphCSV/scenario1lineErrorTimestepBHReverse.csv");
        // writeToFile(timeBHReverseMultipole, errorBHReverseMultipole, "graphCSV/scenario1lineErrorTimestepBHReverseMultipole.csv");
        // writeToFile(timeFMM, errorFMM, "graphCSV/scenario1lineErrorTimestepFMM.csv");
    }

    void errorTimestepFMM()
    {
        int errorMeasurementAmount = 500; // how many iterations to run the simulation
        int particleCount = 1000; // use this many particles

        //generatePoints(particleCount);
        generatePointsCustom1();
        nBodySolversGRAVITY["FMM"]->updateTree(&particles);
        nBodySolversGRAVITY["FMMnaive"]->updateTree(&particles);

        // set graph size
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
            nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeFMM[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMM[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMM[t] /= particles.size();

            nBodySolversGRAVITY["FMMnaive"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            timeFMMnaive[t] = t;
            for (int i = 0; i < particles.size(); i++)
                errorFMMnaive[t] += powf(glm::length(accelerations[i] - accelerationsErrorTest[i]), 2.0f);
            errorFMMnaive[t] /= particles.size();

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;

            //particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

            nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            nBodySolversGRAVITY["FMMnaive"]->updateTree(&particles);
        }

        // write results to csv files
        //writeToFile(timeFMM, errorFMM, "graphCSV/scenario2lineErrorTimestepFMMcompare_FMM.csv");
        //writeToFile(timeFMMnaive, errorFMMnaive, "graphCSV/scenario2lineErrorTimestepFMMcompare_FMMnaive.csv");
    }

    void calculationtimeTheta()
    {
        int averageOverAmount = 1000; // average the error over this many time steps
        int particleCount = 1000; // use this many particles

        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;


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
            generatePoints(particleCount);
            nBodySolversGRAVITY["BH"]->updateTree(&particles);
            nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
            nBodySolversGRAVITY["BHR"]->updateTree(&particles);
            nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
            nBodySolversGRAVITY["FMM"]->updateTree(&particles);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMultipole[t] = chosenTheta;
            thetaBHReverse[t] = chosenTheta;
            thetaBHReverseMultipole[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolversGRAVITY["BH"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHR"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHRMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["FMM"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for(int j = 0; j < averageOverAmount; j++)
            { 
                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);
                calculationtimeNaive[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverse[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHReverseMultipole[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;

                //particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            }

            calculationtimeNaive[t] /= averageOverAmount;
            calculationtimeBH[t] /= averageOverAmount;
            calculationtimeBHMultipole[t] /= averageOverAmount;
            calculationtimeBHReverse[t] /= averageOverAmount;
            calculationtimeBHReverseMultipole[t] /= averageOverAmount;
            calculationtimeFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        // writeToFile(thetaNaive, calculationtimeNaive, ("graphCSV/scenario3calculationtimeThetaNaive.csv"));
        // writeToFile(thetaBH, calculationtimeBH, ("graphCSV/scenario3calculationtimeThetaBH.csv"));
        // writeToFile(thetaBHMultipole, calculationtimeBHMultipole, ("graphCSV/scenario3calculationtimeThetaBHMP.csv"));
        // writeToFile(thetaBHReverse, calculationtimeBHReverse, ("graphCSV/scenario3calculationtimeThetaBHR.csv"));
        // writeToFile(thetaBHReverseMultipole, calculationtimeBHReverseMultipole, ("graphCSV/scenario3calculationtimeThetaBHRMP.csv"));
        // writeToFile(thetaFMM, calculationtimeFMM, ("graphCSV/scenario3calculationtimeThetaFMM.csv"));
    }

    void errorTheta()
    {
        int averageOverAmount = 1000; // average the error over this many time steps
        int particleCount = 1000; // use this many particles

        
        // set graph size
        int thetaDiversityAmount = 100;
        float thetaDiffSize = 2.0f; // size of theta, so 1.0f means that theta will range between (offset, offset + size)
        float thetaOffset = 0.3f;

        

        //std::vector<float> errorNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBH(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMM(thetaDiversityAmount, 0.0f);

        //std::vector<float> thetaNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverse(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHReverseMultipole(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);


        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePoints(particleCount);
            nBodySolversGRAVITY["BH"]->updateTree(&particles);
            nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
            nBodySolversGRAVITY["BHR"]->updateTree(&particles);
            nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
            nBodySolversGRAVITY["FMM"]->updateTree(&particles);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaBH[t] = chosenTheta;
            thetaBHMultipole[t] = chosenTheta;
            thetaBHReverse[t] = chosenTheta;
            thetaBHReverseMultipole[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolversGRAVITY["BH"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHR"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHRMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["FMM"]->theta = chosenTheta;

            for (int j = 0; j < averageOverAmount; j++)
            {
                nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

                nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBH[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBHMultipole[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBHReverse[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBHReverseMultipole[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorFMM[t] += getMSE(accelerations, accelerationsErrorTest);


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;

                //particlesBuffer->updateBuffer(particles, pos2Dvel2Dcol3Dmass);

                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            }

            errorBH[t] /= averageOverAmount;
            errorBHMultipole[t] /= averageOverAmount;
            errorBHReverse[t] /= averageOverAmount;
            errorBHReverseMultipole[t] /= averageOverAmount;
            errorFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        // writeToFile(thetaBH, errorBH, ("graphCSV/scenario4errorThetaBH.csv"));
        // writeToFile(thetaBHMultipole, errorBHMultipole, ("graphCSV/scenario4errorThetaBHMP.csv"));
        // writeToFile(thetaBHReverse, errorBHReverse, ("graphCSV/scenario4errorThetaBHR.csv"));
        // writeToFile(thetaBHReverseMultipole, errorBHReverseMultipole, ("graphCSV/scenario4errorThetaBHRMP.csv"));
        // writeToFile(thetaFMM, errorFMM, ("graphCSV/scenario4errorThetaFMM.csv"));
    }

    void errorTimestepTSNE()
    {
        int errorMeasurementAmount = 500; // how many iterations to run the simulation
        int dataAmount = 1000; // use this many particles
        float perplexity = 30.0f;

        learnRate = 1000.0f;
        accelerationRate = 0.5f;

        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif

        labels = Loader::loadLabels(labelsPath.string());
        Pmatrix = Loader::loadPmatrix(fileName.string());

        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);

        embeddedDerivative.resize(dataAmount);
        embeddedDerivativeErrorTest.resize(dataAmount);
        attractForce.resize(dataAmount);
        repulsForce.resize(dataAmount);
        repulsForceErrorTest.resize(dataAmount);
        repulsForceNotNorm.resize(dataAmount);
        repulsForceErrorTestNotNorm.resize(dataAmount);


        generatePointsTSNE(dataAmount);
        nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
        nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
        nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
        nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
        nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);


        // set graph size
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
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            //lastTimeUpdated = glfwGetTime();
            // correct solution up to machine precision
            updateTSNE("naive", embeddedDerivative, repulsForce);
            //nBodySolversTSNE["naive"]->solveNbody(&noAccumulator, &embeddedDerivative, &particles);

            // calculate the result of every approximation technique and find the error by comparing to naive
            updateTSNE("BH", embeddedDerivativeErrorTest, repulsForceErrorTest);
            timeBH[t] = t;
            for (int i = 0; i < embeddedPoints.size(); i++)
                errorBH[t] += powf(glm::length(repulsForce[i] - repulsForceErrorTest[i]), 2.0f);
            errorBH[t] /= embeddedPoints.size();

            updateTSNE("BHMP", embeddedDerivativeErrorTest, repulsForceErrorTest);
            timeBHMultipole[t] = t;
            for (int i = 0; i < embeddedPoints.size(); i++)
                errorBHMultipole[t] += powf(glm::length(repulsForce[i] - repulsForceErrorTest[i]), 2.0f);
            errorBHMultipole[t] /= embeddedPoints.size();

            updateTSNE("BHR", embeddedDerivativeErrorTest, repulsForceErrorTest);
            timeBHReverse[t] = t;
            for (int i = 0; i < embeddedPoints.size(); i++)
                errorBHReverse[t] += powf(glm::length(repulsForce[i] - repulsForceErrorTest[i]), 2.0f);
            errorBHReverse[t] /= embeddedPoints.size();

            updateTSNE("BHRMP", embeddedDerivativeErrorTest, repulsForceErrorTest);
            timeBHReverseMultipole[t] = t;
            for (int i = 0; i < embeddedPoints.size(); i++)
                errorBHReverseMultipole[t] += powf(glm::length(repulsForce[i] - repulsForceErrorTest[i]), 2.0f);
            errorBHReverseMultipole[t] /= embeddedPoints.size();

            updateTSNE("FMM", embeddedDerivativeErrorTest, repulsForceErrorTest);
            timeFMM[t] = t;
            for (int i = 0; i < embeddedPoints.size(); i++)
                errorFMM[t] += powf(glm::length(repulsForce[i] - repulsForceErrorTest[i]), 2.0f);
            errorFMM[t] /= embeddedPoints.size();

            // update positions with naive solution
            embeddedPointsPrev.swap(embeddedPointsPrevPrev);
            embeddedPoints.swap(embeddedPointsPrev);

            for (int i = 0; i < embeddedPoints.size(); i++)
            {
                embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddedDerivative[i] + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
            }

            nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        writeToFile(timeBH, errorBH, projectFolder / std::filesystem::path("graphCSV") / "scenario5lineErrorTimestepBH.csv");
        writeToFile(timeBHMultipole, errorBHMultipole, projectFolder / std::filesystem::path("graphCSV") / "scenario5lineErrorTimestepBHMultipole.csv");
        writeToFile(timeBHReverse, errorBHReverse, projectFolder / std::filesystem::path("graphCSV") / "scenario5lineErrorTimestepBHReverse.csv");
        writeToFile(timeBHReverseMultipole, errorBHReverseMultipole, projectFolder / std::filesystem::path("graphCSV") / "scenario5lineErrorTimestepBHReverseMultipole.csv");
        writeToFile(timeFMM, errorFMM, projectFolder / std::filesystem::path("graphCSV") / "scenario5lineErrorTimestepFMM.csv");
    }

private:
    float getMSE(const std::vector<glm::vec2>& MSEaccelerations, const std::vector<glm::vec2>& MSEaccelerationsErrorTest)
    {
        float MSE = 0.0f;

        for (int i = 0; i < MSEaccelerations.size(); i++)
        {
            MSE += powf(glm::length(MSEaccelerations[i] - MSEaccelerationsErrorTest[i]), 2.0f);
        }

        MSE /= MSEaccelerations.size();
        return MSE;
    }

    // template <typename T, typename I>
    // void writeToFile(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, std::string filename)
    // {
    //     std::ofstream file(filename);
    //     for (size_t i = 0; i < Xaxis.size(); ++i)
    //         file << Xaxis[i] << "," << Yaxis[i] << "\n";
    // }

    template <typename T, typename I>
    void writeToFile(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);

        if (!file.is_open()) 
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "\n";

        if (file.fail()) 
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else 
            std::cout << "File written successfully: " << filepath << std::endl;

        file.close();
    }

    void generatePoints(int particleAmount)
    {
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        srand(1952731);
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

        //particlesBuffer = new Buffer(particles, pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
    }

    void generatePointsTSNE(int particleAmount)
    {
        srand(1952732);
        float sizeParam = 2.0f;
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
                powf(sizeParam * randX, 1.0f),
                powf(sizeParam * randY, 1.0f)
            );

            int lab = labels[i];

            embeddedPoints[i] = EmbeddedPoint(pos, lab);
            embeddedPointsPrev[i] = EmbeddedPoint(pos, lab);
            embeddedPointsPrevPrev[i] = EmbeddedPoint(pos, lab);
        }
    }

    //void updateAttractive()
    //{
    //    std::fill(attractForce.begin(), attractForce.end(), glm::vec2(0.0f, 0.0f));

    //    for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
    //    {
    //        for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
    //        {
    //            glm::vec2 diff = embeddedPoints[it.col()].position - embeddedPoints[it.row()].position;
    //            float distance = glm::length(diff);

    //            attractForce[it.col()] += -(float)it.value() * (diff / (1.0f + distance));
    //        }
    //    }
    //}

    void updateTSNE(std::string nbodySolverName, std::vector<glm::vec2>& derivResult, std::vector<glm::vec2>& repulResult)
    {
        float QijTotal = 0.0f;

        nBodySolversTSNE[nbodySolverName]->solveNbody(&QijTotal, &repulResult, &embeddedPoints);

        

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            repulResult[i] *= (1.0f / QijTotal);
        }

        // --------------------------------

        std::fill(attractForce.begin(), attractForce.end(), glm::vec2(0.0f, 0.0f));

        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                glm::vec2 diff = embeddedPoints[it.col()].position - embeddedPoints[it.row()].position;
                float distance = glm::length(diff);

                attractForce[it.col()] += -(float)it.value() * (diff / (1.0f + distance));
            }
        }

        // --------------------------------

        std::fill(derivResult.begin(), derivResult.end(), glm::vec2(0.0f, 0.0f));
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            derivResult[i] = attractForce[i] - repulResult[i];
        }
    }

    void generatePointsCustom1()
    {
        int particleAmount = 40;
        particles.resize(particleAmount);
        accelerations.resize(particleAmount);
        accelerationsErrorTest.resize(particleAmount);

        std::vector<glm::vec2> positions;

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
        for (int i = 0; i < 10; i++) { positions[i] += glm::vec2(2.0f, 2.0f); }

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
        for (int i = 10; i < 20; i++) { positions[i] += glm::vec2(2.0f, -2.0f); }

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
        for (int i = 20; i < 30; i++) { positions[i] += glm::vec2(-2.0f, 2.0f); }

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
        for (int i = 30; i < 40; i++) { positions[i] += glm::vec2(-2.0f, -2.0f); }

        float velParam = 0.2f;
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

        //particlesBuffer = new Buffer(particles, pos2Dvel2Dcol3Dmass, GL_DYNAMIC_DRAW);
    }

    void smartNodeNode(std::vector<Particle2D>& particles, std::vector<glm::vec2>& smartNodeNodeAccelerations, glm::vec2 centreOfMass1, float totalMass1, glm::vec2 dipole1, Fastor::Tensor<float, 2, 2> quadrupole1, glm::vec2 centreOfMass2, float totalMass2, glm::vec2 dipole2, Fastor::Tensor<float, 2, 2> quadrupole2, bool firstHalve)
    {
        // prework
        float softening = 0.1f;

        glm::vec2 R = centreOfMass1 - centreOfMass2; // dhenen
        float r = glm::length(R);
        float rS = r + softening;

        float D1 = -1.0f / (rS * rS * rS);
        float D2 = 3.0f / (rS * rS * rS * rS * rS);
        float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);

        float MA0 = totalMass1;
        float MB0 = totalMass2;
        Fastor::Tensor<float, 2, 2> MB2 = quadrupole2;
        Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

        // calculate the C^m
        float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
        float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
        float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
        float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

        Fastor::Tensor<float, 2> C1 =
        {
            MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
            MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
        };

        Fastor::Tensor<float, 2, 2> C2 =
        {
            {
                MB0 * (D1 + R.x * R.x * D2),
                MB0 * (R.x * R.y * D2)
            },
            {
                MB0 * (R.y * R.x * D2),
                MB0 * (D1 + R.y * R.y * D2)
            }
        };

        Fastor::Tensor<float, 2, 2, 2> C3 =
        {
            {
                {
                    MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                    MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
                },
                {
                    MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                    MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
                }
            },
            {
                {
                    MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                    MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
                },
                {
                    MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                    MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
                }
            }
        };

        //passiveNode->C1 += C1;
        //passiveNode->C2 += C2;
        //passiveNode->C3 += C3;

        int start = 10;
        if (firstHalve)
        {
            start = 0;
        }


        for (int i = start; i < start + 10; i++)
        {
            glm::vec2 x = particles[i].position;
            glm::vec2 Z0 = centreOfMass1;
            Fastor::Tensor<float, 2> diff1 = { x.x - Z0.x, x.y - Z0.y };
            Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
            Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

            // evaluate C^n at occupants position then add to occupant acceleration // might be wrong!!!!!!!!!!!
            Fastor::Tensor<float, 2> acceleration = C1 +
                Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
                //(1.0f / 2.0f) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);
                (1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));

            smartNodeNodeAccelerations[i] += glm::vec2(acceleration(0), acceleration(1));
        }

    }
};