#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
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
    // GRAVITY variables
    std::vector<Particle2D> particles;
    std::vector<glm::vec2> accelerations;
    std::vector<glm::vec2> accelerationsErrorTest;

    float stepSize = 0.1f;
    float timeStepsPerSec = 30.0f;
    float lastTimeUpdated = 0.0f;

    std::map<std::string, NBodySolver<Particle2D>*> nBodySolversGRAVITY;



    // TSNE variables
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

    float learnRate;
    float accelerationRate;
    float perplexity;
    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;

    std::map<std::string, NBodySolver<EmbeddedPoint>*> nBodySolversTSNE;


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
        //nBodySolversTSNE["FMMiter"] = new NBodySolverFMMiter<EmbeddedPoint>(&TSNEFMMiterInteractionKernal, 10, 1.0f);
        nBodySolversTSNE["FMMiter"] = new NBodySolverFMMiter<EmbeddedPoint>(&TSNEFMMiterInteractionKernalNodeNode, &TSNEFMMiterInteractionKernalNodeParticle, &TSNEFMMiterInteractionKernalParticleNode, &TSNEFMMiterInteractionKernalParticleParticle, 10, 1.0f);
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



    void errorTimestepGRAVITY()
    {
        int errorMeasurementAmount = 1000; // how many iterations to run the simulation
        int particleCount = 1000; // use this many particles
        float setTheta = 1.0f; // approximation parameter

        generatePoints(particleCount);
        nBodySolversGRAVITY["BH"]->updateTree(&particles);
        nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
        nBodySolversGRAVITY["BHR"]->updateTree(&particles);
        nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
        nBodySolversGRAVITY["FMM"]->updateTree(&particles);

        nBodySolversGRAVITY["BH"]->theta = setTheta;
        nBodySolversGRAVITY["BHMP"]->theta = setTheta;
        nBodySolversGRAVITY["BHR"]->theta = setTheta;
        nBodySolversGRAVITY["BHRMP"]->theta = setTheta;
        nBodySolversGRAVITY["FMM"]->theta = setTheta;


        // set graph size
        std::vector<float> errorBH(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHMP(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHR(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHRMP(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMM(errorMeasurementAmount, 0.0f);

        std::vector<int> timeBH(errorMeasurementAmount, 0);
        std::vector<int> timeBHMP(errorMeasurementAmount, 0);
        std::vector<int> timeBHR(errorMeasurementAmount, 0);
        std::vector<int> timeBHRMP(errorMeasurementAmount, 0);
        std::vector<int> timeFMM(errorMeasurementAmount, 0);


        // find error at every time step
        float noAccumulator = 0.0f;
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            // correct solution up to machine precision
            nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);


            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorBH[t] = getMSE(accelerations, accelerationsErrorTest);
            timeBH[t] = t;
            
            nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorBHMP[t] = getMSE(accelerations, accelerationsErrorTest);
            timeBHMP[t] = t;

            nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorBHR[t] = getMSE(accelerations, accelerationsErrorTest);
            timeBHR[t] = t;

            nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorBHRMP[t] = getMSE(accelerations, accelerationsErrorTest);
            timeBHRMP[t] = t;

            nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);//, 10, 1.0f
            errorFMM[t] = getMSE(accelerations, accelerationsErrorTest);
            timeFMM[t] = t;

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;
            
            // update tree structures
            nBodySolversGRAVITY["BH"]->updateTree(&particles);
            nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
            nBodySolversGRAVITY["BHR"]->updateTree(&particles);
            nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
            nBodySolversGRAVITY["FMM"]->updateTree(&particles);
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(particleCount) + "_theta" + fltToStr(setTheta);
        writeToFile(timeBH,    errorBH,    projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepBH"    + attributes + ".csv"));
        writeToFile(timeBHMP,  errorBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepBHMP"  + attributes + ".csv"));
        writeToFile(timeBHR,   errorBHR,   projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepBHR"   + attributes + ".csv"));
        writeToFile(timeBHRMP, errorBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepBHRMP" + attributes + ".csv"));
        writeToFile(timeFMM,   errorFMM,   projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepFMM"   + attributes + ".csv"));
    }

    void errorTimestepTSNE()
    {
        int errorMeasurementAmount = 500; // how many iterations to run the simulation
        int dataAmount = 1000; // use this many particles
        float setTheta = 1.0f; // approximation parameter

        perplexity = 30.0f;
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
        nBodySolversTSNE["FMMiter"]->updateTree(&embeddedPoints);

        nBodySolversTSNE["BH"]->theta = setTheta;
        nBodySolversTSNE["BHMP"]->theta = setTheta;
        nBodySolversTSNE["BHR"]->theta = setTheta;
        nBodySolversTSNE["BHRMP"]->theta = setTheta;
        nBodySolversTSNE["FMM"]->theta = setTheta;
        nBodySolversTSNE["FMMiter"]->theta = setTheta;


        // set graph size
        std::vector<float> errorBH(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHMP(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHR(errorMeasurementAmount, 0.0f);
        std::vector<float> errorBHRMP(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMM(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMMiter(errorMeasurementAmount, 0.0f);

        std::vector<int> timeBH(errorMeasurementAmount, 0);
        std::vector<int> timeBHMP(errorMeasurementAmount, 0);
        std::vector<int> timeBHR(errorMeasurementAmount, 0);
        std::vector<int> timeBHRMP(errorMeasurementAmount, 0);
        std::vector<int> timeFMM(errorMeasurementAmount, 0);
        std::vector<int> timeFMMiter(errorMeasurementAmount, 0);


        // find error at every time step
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            // correct solution up to machine precision
            updateTSNE("naive", embeddedDerivative, repulsForce, repulsForceNotNorm);

            // calculate the result of every approximation technique and find the error by comparing to naive
            updateTSNE("BH", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorBH[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeBH[t] = t;

            updateTSNE("BHMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorBHMP[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeBHMP[t] = t;

            updateTSNE("BHR", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorBHR[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeBHR[t] = t;

            updateTSNE("BHRMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorBHRMP[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeBHRMP[t] = t;

            updateTSNE("FMM", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorFMM[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeFMM[t] = t;

            updateTSNE("FMMiter", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
            errorFMMiter[t] = getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);
            timeFMMiter[t] = t;


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
            nBodySolversTSNE["FMMiter"]->updateTree(&embeddedPoints);
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(dataAmount) + "_theta" + fltToStr(setTheta);
        writeToFile(timeBH,      errorBH,      projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepBH"      + attributes + ".csv"));
        writeToFile(timeBHMP,    errorBHMP,    projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepBHMP"    + attributes + ".csv"));
        writeToFile(timeBHR,     errorBHR,     projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepBHR"     + attributes + ".csv"));
        writeToFile(timeBHRMP,   errorBHRMP,   projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepBHRMP"   + attributes + ".csv"));
        writeToFile(timeFMM,     errorFMM,     projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepFMM"     + attributes + ".csv"));
        writeToFile(timeFMMiter, errorFMMiter, projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorTimestepFMMiter" + attributes + ".csv"));
    }


    void errorTimestepGRAVITYFMMtest()
    {
        int errorMeasurementAmount = 500; // how many iterations to run the simulation
        int particleCount = 500; // use this many particles
        float setTheta = 1.0f; // approximation parameter

        //generatePoints(particleCount);
        generatePointsCustom1();
        nBodySolversGRAVITY["FMM"]->updateTree(&particles);
        nBodySolversGRAVITY["FMMnaive"]->updateTree(&particles);

        nBodySolversGRAVITY["FMM"]->theta = setTheta;
        nBodySolversGRAVITY["FMMnaive"]->theta = setTheta;

        // set graph size
        std::vector<float> errorFMM(errorMeasurementAmount, 0.0f);
        std::vector<float> errorFMMnaive(errorMeasurementAmount, 0.0f);

        std::vector<int> timeFMM(errorMeasurementAmount, 0);
        std::vector<int> timeFMMnaive(errorMeasurementAmount, 0);


        // find error at every time step
        float noAccumulator = 0.0f;
        for (int t = 0; t < errorMeasurementAmount; t++)
        {
            // correct solution up to machine precision
            nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);


            // calculate the result of every approximation technique and find the error by comparing to naive
            nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorFMM[t] = getMSE(accelerations, accelerationsErrorTest);
            timeFMM[t] = t;

            nBodySolversGRAVITY["FMMnaive"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
            errorFMMnaive[t] = getMSE(accelerations, accelerationsErrorTest);
            timeFMMnaive[t] = t;

            // update positions with naive solution
            for (int i = 0; i < particles.size(); i++)
                particles[i].speed += stepSize * accelerations[i];

            for (int i = 0; i < particles.size(); i++)
                particles[i].position += stepSize * particles[i].speed;


            nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            nBodySolversGRAVITY["FMMnaive"]->updateTree(&particles);
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(particleCount) + "_theta" + fltToStr(setTheta);   
        writeToFile(timeFMM, errorFMM, projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepCompareFMM" + attributes + ".csv"));
        writeToFile(timeFMMnaive, errorFMMnaive, projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorTimestepCompareFMMnaive" + attributes + ".csv"));
    }


    void calculationtimeThetaGRAVITY()
    {
        int averageOverAmount = 500; // average the error over this many time steps
        int particleCount = 1000; // use this many particles

        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;


        std::vector<float> calculationtimeNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
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
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
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
                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;

                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            }

            calculationtimeNaive[t] /= averageOverAmount;
            calculationtimeBH[t]    /= averageOverAmount;
            calculationtimeBHMP[t]  /= averageOverAmount;
            calculationtimeBHR[t]   /= averageOverAmount;
            calculationtimeBHRMP[t] /= averageOverAmount;
            calculationtimeFMM[t]   /= averageOverAmount;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(particleCount);
        writeToFile(thetaNaive, calculationtimeNaive, projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaNaive" + attributes + ".csv"));
        writeToFile(thetaBH,    calculationtimeBH,    projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaBH" + attributes + ".csv"));
        writeToFile(thetaBHMP,  calculationtimeBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaBHMP" + attributes + ".csv"));
        writeToFile(thetaBHR,   calculationtimeBHR,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaBHR" + attributes + ".csv"));
        writeToFile(thetaBHRMP, calculationtimeBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaBHRMP" + attributes + ".csv"));
        writeToFile(thetaFMM,   calculationtimeFMM,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeThetaFMM" + attributes + ".csv"));
    }

    void calculationtimeThetaTSNE()
    {
        int averageOverAmount = 500; // average the error over this many time steps
        int dataAmount = 1000; // use this many particles

        // set graph size
        int thetaDiversityAmount = 10;
        float thetaDiffSize = 1.0f;
        float thetaOffset = 0.3f;


        perplexity = 30.0f;
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


        std::vector<float> calculationtimeNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaNaive(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);


        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePointsTSNE(dataAmount);
            nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolversTSNE["BH"]->theta = chosenTheta;
            nBodySolversTSNE["BHMP"]->theta = chosenTheta;
            nBodySolversTSNE["BHR"]->theta = chosenTheta;
            nBodySolversTSNE["BHRMP"]->theta = chosenTheta;
            nBodySolversTSNE["FMM"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for(int j = 0; j < averageOverAmount; j++)
            { 
                timeBefore = glfwGetTime();
                updateTSNE("naive", embeddedDerivative, repulsForce, repulsForceNotNorm);
                calculationtimeNaive[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
                updateTSNE("BH", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
                updateTSNE("BHMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
                updateTSNE("BHR", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
                updateTSNE("BHRMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;

                timeBefore = glfwGetTime();
                nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);
                updateTSNE("FMM", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;


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

            calculationtimeNaive[t] /= averageOverAmount;
            calculationtimeBH[t]    /= averageOverAmount;
            calculationtimeBHMP[t]  /= averageOverAmount;
            calculationtimeBHR[t]   /= averageOverAmount;
            calculationtimeBHRMP[t] /= averageOverAmount;
            calculationtimeFMM[t]   /= averageOverAmount;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(dataAmount);
        writeToFile(thetaNaive, calculationtimeNaive, projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaNaive" + attributes + ".csv"));
        writeToFile(thetaBH,    calculationtimeBH,    projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaBH" + attributes + ".csv"));
        writeToFile(thetaBHMP,  calculationtimeBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaBHMP" + attributes + ".csv"));
        writeToFile(thetaBHR,   calculationtimeBHR,   projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaBHR" + attributes + ".csv"));
        writeToFile(thetaBHRMP, calculationtimeBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaBHRMP" + attributes + ".csv"));
        writeToFile(thetaFMM,   calculationtimeFMM,   projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeThetaFMM" + attributes + ".csv"));
    }


    void errorThetaGRAVITY()
    {
        int averageOverAmount = 250; // average the error over this many time steps
        int particleCount = 1000; // use this many particles

        
        // set graph size
        int thetaDiversityAmount = 100;
        float thetaDiffSize = 2.0f; // size of theta, so 1.0f means that theta will range between (offset, offset + size)
        float thetaOffset = 0.3f;

        
        std::vector<float> errorBH(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
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
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
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
                errorBHMP[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBHR[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorBHRMP[t] += getMSE(accelerations, accelerationsErrorTest);

                nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                errorFMM[t] += getMSE(accelerations, accelerationsErrorTest);


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;


                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            }

            errorBH[t] /= averageOverAmount;
            errorBHMP[t] /= averageOverAmount;
            errorBHR[t] /= averageOverAmount;
            errorBHRMP[t] /= averageOverAmount;
            errorFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(particleCount);
        writeToFile(thetaBH,    errorBH,    projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorThetaBH" + attributes + ".csv"));
        writeToFile(thetaBHMP,  errorBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorThetaBHMP" + attributes + ".csv"));
        writeToFile(thetaBHR,   errorBHR,   projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorThetaBHR" + attributes + ".csv"));
        writeToFile(thetaBHRMP, errorBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorThetaBHRMP" + attributes + ".csv"));
        writeToFile(thetaFMM,   errorFMM,   projectFolder / std::filesystem::path("graphCSV") / ("gravityErrorThetaFMM" + attributes + ".csv"));
    }

    void errorThetaTSNE()
    {
        int averageOverAmount = 250; // average the error over this many time steps
        int dataAmount = 1000; // use this many particles

        
        // set graph size
        int thetaDiversityAmount = 100;
        float thetaDiffSize = 2.0f; // size of theta, so 1.0f means that theta will range between (offset, offset + size)
        float thetaOffset = 0.3f;


        perplexity = 30.0f;
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

        
        std::vector<float> errorBH(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);


        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePointsTSNE(dataAmount);
            nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;

            nBodySolversTSNE["BH"]->theta = chosenTheta;
            nBodySolversTSNE["BHMP"]->theta = chosenTheta;
            nBodySolversTSNE["BHR"]->theta = chosenTheta;
            nBodySolversTSNE["BHRMP"]->theta = chosenTheta;
            nBodySolversTSNE["FMM"]->theta = chosenTheta;

            for (int j = 0; j < averageOverAmount; j++)
            {
                updateTSNE("naive", embeddedDerivative, repulsForce, repulsForceNotNorm);


                updateTSNE("BH", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                errorBH[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                updateTSNE("BHMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                errorBHMP[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                updateTSNE("BHR", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                errorBHR[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                updateTSNE("BHRMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                errorBHRMP[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                updateTSNE("FMM", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                errorFMM[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);


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

            errorBH[t] /= averageOverAmount;
            errorBHMP[t] /= averageOverAmount;
            errorBHR[t] /= averageOverAmount;
            errorBHRMP[t] /= averageOverAmount;
            errorFMM[t] /= averageOverAmount;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(dataAmount);
        writeToFile(thetaBH,    errorBH,    projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorThetaBH" + attributes + ".csv"));
        writeToFile(thetaBHMP,  errorBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorThetaBHMP" + attributes + ".csv"));
        writeToFile(thetaBHR,   errorBHR,   projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorThetaBHR" + attributes + ".csv"));
        writeToFile(thetaBHRMP, errorBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorThetaBHRMP" + attributes + ".csv"));
        writeToFile(thetaFMM,   errorFMM,   projectFolder / std::filesystem::path("graphCSV") / ("tsneErrorThetaFMM" + attributes + ".csv"));
    }


    void calculationtimeErrorGRAVITY()
    {
        int averageOverAmount = 150; // average the error over this many time steps
        int particleCount = 1000; // use this many particles

        
        // set graph size
        int thetaDiversityAmount = 40;
        float thetaDiffSize = 2.0f; // size of theta, so 1.0f means that theta will range between (offset, offset + size)
        float thetaOffset = 0.3f;

        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);
        
        std::vector<float> errorBH(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMM(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
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

            nBodySolversGRAVITY["BH"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHR"]->theta = chosenTheta;
            nBodySolversGRAVITY["BHRMP"]->theta = chosenTheta;
            nBodySolversGRAVITY["FMM"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for (int j = 0; j < averageOverAmount; j++)
            {
                nBodySolversGRAVITY["naive"]->solveNbody(&noAccumulator, &accelerations, &particles);

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BH"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
                errorBH[t] += getMSE(accelerations, accelerationsErrorTest);

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
                errorBHMP[t] += getMSE(accelerations, accelerationsErrorTest);

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;
                errorBHR[t] += getMSE(accelerations, accelerationsErrorTest);

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;
                errorBHRMP[t] += getMSE(accelerations, accelerationsErrorTest);

                timeBefore = glfwGetTime();
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->solveNbody(&noAccumulator, &accelerationsErrorTest, &particles);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;
                errorFMM[t] += getMSE(accelerations, accelerationsErrorTest);


                // update positions with naive solution
                for (int i = 0; i < particles.size(); i++)
                    particles[i].speed += stepSize * accelerations[i];

                for (int i = 0; i < particles.size(); i++)
                    particles[i].position += stepSize * particles[i].speed;


                nBodySolversGRAVITY["BH"]->updateTree(&particles);
                nBodySolversGRAVITY["BHMP"]->updateTree(&particles);
                nBodySolversGRAVITY["BHR"]->updateTree(&particles);
                nBodySolversGRAVITY["BHRMP"]->updateTree(&particles);
                nBodySolversGRAVITY["FMM"]->updateTree(&particles);
            }

            errorBH[t] /= averageOverAmount;
            errorBHMP[t] /= averageOverAmount;
            errorBHR[t] /= averageOverAmount;
            errorBHRMP[t] /= averageOverAmount;
            errorFMM[t] /= averageOverAmount;

            calculationtimeBH[t] /= averageOverAmount;
            calculationtimeBHMP[t] /= averageOverAmount;
            calculationtimeBHR[t] /= averageOverAmount;
            calculationtimeBHRMP[t] /= averageOverAmount;
            calculationtimeFMM[t] /= averageOverAmount;

            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(particleCount);

        writeToFile3(errorBH,    calculationtimeBH,    thetaBH,    projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBH" + attributes + ".csv"));
        writeToFile3(errorBHMP,  calculationtimeBHMP,  thetaBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHMP" + attributes + ".csv"));
        writeToFile3(errorBHR,   calculationtimeBHR,   thetaBHR,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHR" + attributes + ".csv"));
        writeToFile3(errorBHRMP, calculationtimeBHRMP, thetaBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHRMP" + attributes + ".csv"));
        writeToFile3(errorFMM,   calculationtimeFMM,   thetaFMM,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorFMM" + attributes + ".csv"));

        //writeToFile(errorBH,    calculationtimeBH,    projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBH" + attributes + ".csv"));
        //writeToFile(errorBHMP,  calculationtimeBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHMP" + attributes + ".csv"));
        //writeToFile(errorBHR,   calculationtimeBHR,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHR" + attributes + ".csv"));
        //writeToFile(errorBHRMP, calculationtimeBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorBHRMP" + attributes + ".csv"));
        //writeToFile(errorFMM,   calculationtimeFMM,   projectFolder / std::filesystem::path("graphCSV") / ("gravityCalculationtimeErrorFMM" + attributes + ".csv"));
    }

    void calculationtimeErrorTSNE()
    {
        int averageOverAmount = 50; // average the error over this many time steps
        int dataAmount = 1000; // use this many particles

        
        // set graph size
        int thetaDiversityAmount = 20;
        float thetaDiffSize = 2.0f; // size of theta, so 1.0f means that theta will range between (offset, offset + size)
        float thetaOffset = 0.3f;


        perplexity = 30.0f;
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


        std::vector<float> calculationtimeBH(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMM(thetaDiversityAmount, 0.0f);
        std::vector<float> calculationtimeFMMiter(thetaDiversityAmount, 0.0f);
        
        std::vector<float> errorBH(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> errorBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMM(thetaDiversityAmount, 0.0f);
        std::vector<float> errorFMMiter(thetaDiversityAmount, 0.0f);

        std::vector<float> thetaBH(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHR(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaBHRMP(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMM(thetaDiversityAmount, 0.0f);
        std::vector<float> thetaFMMiter(thetaDiversityAmount, 0.0f);

        float noAccumulator = 0.0f;
        for (int t = 0; t < thetaDiversityAmount; t++)
        {
            generatePointsTSNE(dataAmount);
            nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);
            nBodySolversTSNE["FMMiter"]->updateTree(&embeddedPoints);

            float chosenTheta = ((float)t / thetaDiversityAmount) * thetaDiffSize + thetaOffset;

            nBodySolversTSNE["BH"]->theta = chosenTheta;
            nBodySolversTSNE["BHMP"]->theta = chosenTheta;
            nBodySolversTSNE["BHR"]->theta = chosenTheta;
            nBodySolversTSNE["BHRMP"]->theta = chosenTheta;
            nBodySolversTSNE["FMM"]->theta = chosenTheta;
            nBodySolversTSNE["FMMiter"]->theta = chosenTheta;

            float timeBefore = 0.0f;
            for (int j = 0; j < averageOverAmount; j++)
            {
                updateTSNE("naive", embeddedDerivative, repulsForce, repulsForceNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BH"]->updateTree(&embeddedPoints);
                updateTSNE("BH", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
                errorBH[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHMP"]->updateTree(&embeddedPoints);
                updateTSNE("BHMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
                errorBHMP[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHR"]->updateTree(&embeddedPoints);
                updateTSNE("BHR", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;
                errorBHR[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["BHRMP"]->updateTree(&embeddedPoints);
                updateTSNE("BHRMP", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;
                errorBHRMP[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["FMM"]->updateTree(&embeddedPoints);
                updateTSNE("FMM", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;
                errorFMM[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);

                timeBefore = glfwGetTime();
                nBodySolversTSNE["FMMiter"]->updateTree(&embeddedPoints);
                updateTSNE("FMMiter", embeddedDerivativeErrorTest, repulsForceErrorTest, repulsForceErrorTestNotNorm);
                calculationtimeFMMiter[t] += glfwGetTime() - timeBefore;
                errorFMMiter[t] += getMSE(repulsForceNotNorm, repulsForceErrorTestNotNorm);


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
                nBodySolversTSNE["FMMiter"]->updateTree(&embeddedPoints);
            }

            errorBH[t] /= averageOverAmount;
            errorBHMP[t] /= averageOverAmount;
            errorBHR[t] /= averageOverAmount;
            errorBHRMP[t] /= averageOverAmount;
            errorFMM[t] /= averageOverAmount;
            errorFMMiter[t] /= averageOverAmount;

            calculationtimeBH[t] /= averageOverAmount;
            calculationtimeBHMP[t] /= averageOverAmount;
            calculationtimeBHR[t] /= averageOverAmount;
            calculationtimeBHRMP[t] /= averageOverAmount;
            calculationtimeFMM[t] /= averageOverAmount;
            calculationtimeFMMiter[t] /= averageOverAmount;

            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaFMMiter[t] = chosenTheta;
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        std::string attributes = "_point" + std::to_string(dataAmount);

        writeToFile3(errorBH,      calculationtimeBH,      thetaBH,      projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBH"      + attributes + ".csv"));
        writeToFile3(errorBHMP,    calculationtimeBHMP,    thetaBHMP,    projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHMP"    + attributes + ".csv"));
        writeToFile3(errorBHR,     calculationtimeBHR,     thetaBHR,     projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHR"     + attributes + ".csv"));
        writeToFile3(errorBHRMP,   calculationtimeBHRMP,   thetaBHRMP,   projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHRMP"   + attributes + ".csv"));
        writeToFile3(errorFMM,     calculationtimeFMM,     thetaFMM,     projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorFMM"     + attributes + ".csv"));
        writeToFile3(errorFMMiter, calculationtimeFMMiter, thetaFMMiter, projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorFMMiter" + attributes + ".csv"));

        //writeToFileN(projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorFMM" + attributes + ".csv"), errorFMM, calculationtimeFMM, thetaFMM);

        //writeToFile(errorBH,    calculationtimeBH,    projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBH" + attributes + ".csv"));
        //writeToFile(errorBHMP,  calculationtimeBHMP,  projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHMP" + attributes + ".csv"));
        //writeToFile(errorBHR,   calculationtimeBHR,   projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHR" + attributes + ".csv"));
        //writeToFile(errorBHRMP, calculationtimeBHRMP, projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorBHRMP" + attributes + ".csv"));
        //writeToFile(errorFMM,   calculationtimeFMM,   projectFolder / std::filesystem::path("graphCSV") / ("tsneCalculationtimeErrorFMM" + attributes + ".csv"));
    }

private:
    float getMSE(const std::vector<glm::vec2>& MSEaccelerations, const std::vector<glm::vec2>& MSEaccelerationsErrorTest)
    {
        //float MSE = 0.0f;

        //for (int i = 0; i < MSEaccelerations.size(); i++)
        //{
        //    MSE += powf(glm::length(MSEaccelerations[i] - MSEaccelerationsErrorTest[i]), 2.0f);
        //}

        //MSE /= MSEaccelerations.size();
        //return MSE;

        // ------------------------------------------------------------

        float MSE = 0.0f;
        float divide = 0.0f;

        for (int i = 0; i < MSEaccelerations.size(); i++)
        {
            MSE += powf(glm::length(MSEaccelerations[i] - MSEaccelerationsErrorTest[i]), 1.0f);
            divide += powf(glm::length(MSEaccelerations[i]), 1.0f);
        }

        float NMSE =  MSE / divide;
        return NMSE;
    }

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

    template <typename T, typename I, typename J>
    void writeToFile3(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, const std::vector<J>& Zaxis, std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "," << Zaxis[i] << "\n";

        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;

        file.close();
    }

    template <typename... Vectors>
    void writeToFileN(const std::filesystem::path& filepath, const Vectors&... vectors)
    {
        std::ofstream file(filepath, std::ios::out);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }

        size_t size = std::min({ vectors.size()... }); // take smallest size of vectors

        for (size_t i = 0; i < size; ++i)
        {
            std::ostringstream oss;
            ((oss << vectors[i] << ","), ...);
            std::string line = oss.str();
            file << line.substr(0, line.size() - 1) << "\n";

            //((file << vectors[i] << ((&vectors == &std::get<sizeof...(vectors) - 1>(std::tie(vectors...))) ? "\n" : ",")), ...);
        }

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

    void updateTSNE(std::string nbodySolverName, std::vector<glm::vec2>& derivResult, std::vector<glm::vec2>& repulResult, std::vector<glm::vec2>& repulNotNormResult)
    {
        std::fill(repulResult.begin(), repulResult.end(), glm::vec2(0.0f, 0.0f));
        std::fill(repulNotNormResult.begin(), repulNotNormResult.end(), glm::vec2(0.0f, 0.0f));

        float QijTotal = 0.0f;

        nBodySolversTSNE[nbodySolverName]->solveNbody(&QijTotal, &repulResult, &embeddedPoints);
        for (int i = 0; i < repulResult.size(); i++)
        { 
            repulNotNormResult[i] = repulResult[i];
        }

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

    std::string fltToStr(float f) 
    {
        std::ostringstream oss;
        oss << std::fixed << f;
        std::string str = oss.str();

        // remove zero's
        str.erase(str.find_last_not_of('0') + 1);

        // If it ends with a . remove it too
        if (!str.empty() && str.back() == '.') 
        {
            str.pop_back();
        }

        return str;
    }
};