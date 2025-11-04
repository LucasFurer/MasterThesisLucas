#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <unsupported/Eigen/SparseExtra>
#include <limits>

#include "../particles/embeddedPoint.h"
#include "../openGLhelper/buffer.h"
#include "../common.h"
#include "../openGLhelper/buffer.h"
#include "../dataLoaders/loader.h"
#include "../nbodysolvers/cpu/nBodySolverNaive.h"
#include "../nbodysolvers/cpu/nBodySolverBarnesHut.h"
#include "../nbodysolvers/cpu/nBodySolverBarnesHutReverse.h"
#include "../nbodysolvers/cpu/nBodySolverBarnesHutReverseMultiPole.h"
#include "../nbodysolvers/cpu/nBodySolverMultiPole.h"
#include "../nbodysolvers/cpu/nBodySolverFMM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"


class TSNE
{
public:
    std::vector<EmbeddedPoint> embeddedPoints;
    std::vector<EmbeddedPoint> embeddedPointsPrev;
    std::vector<EmbeddedPoint> embeddedPointsPrevPrev;
    Buffer* embeddedBuffer;

    Buffer* forceBuffer;
    float forceSize = 1.0f;

    std::vector<glm::vec2> embeddedDerivative;
    std::vector<glm::vec2> attractForce;
    std::vector<glm::vec2> repulsForce;

    std::vector<glm::vec2> errorCompare;

    std::map<std::string, NBodySolver<EmbeddedPoint>*> nBodySolvers;
    std::string nBodySelect = "naive";

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec;
    float lastTimeUpdated;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    //std::vector<std::vector<float>> Qmatrix;
    float Qsum;

    int follow = 1;
    float totalError = 0.0f;
    int timeCounter = 0;

    float totalError1 = 0.0f;
    float totalError2 = 0.0f;
    float maxError1 = 0.0f;
    float maxError2 = 0.0f;
    int globalTimeStep = 0;
    
	TSNE()
	{
        int dataAmount = 10000;
        float perplexity = 30.0f;
        std::string dataSet = "MNIST_digits";
        //std::string dataSet = "CIFAR10";

        learnRate = 1000.0f;
        //accelerationRate = 0.5f;

        timeStepsPerSec = 0.0f;

        lastTimeUpdated = 0.0f;

        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/" + dataSet + "/" + std::to_string(dataAmount) + "/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/" + dataSet + "/" + std::to_string(dataAmount) + "/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif

        labels = Loader::loadLabels(labelsPath.string());
        Pmatrix = Loader::loadPmatrix(fileName.string());

        //Qmatrix.resize(dataAmount);
        //for (int i = 0; i < dataAmount; i++) { Qmatrix[i].resize(dataAmount); }

        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);

        embeddedDerivative.resize(dataAmount);
        attractForce.resize(dataAmount);
        repulsForce.resize(dataAmount);

        errorCompare.resize(dataAmount);

        //srand(time(NULL));
        srand(1952732);
        float sizeParam = 2.0f;
        for (int i = 0; i < dataAmount; i++)
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

        nBodySolvers["naive"] = new NBodySolverNaive<EmbeddedPoint>(&TSNEnaiveKernal);
        nBodySolvers["BH"] = new NBodySolverBarnesHut<EmbeddedPoint>(&TSNEbarnesHutParticleNodeKernal, &TSNEbarnesHutParticleParticleKernal, 10, 1.0f); 
        nBodySolvers["BH"]->updateTree(&embeddedPoints);
        nBodySolvers["BHR"] = new NBodySolverBarnesHutReverse<EmbeddedPoint>(&TSNEbarnesHutReverseParticleNodeKernal, &TSNEbarnesHutReverseParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHR"]->updateTree(&embeddedPoints);
        nBodySolvers["BHMP"] = new NBodySolverMultiPole<EmbeddedPoint>(&TSNEmultiPoleParticleNodeKernal, &TSNEmultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHMP"]->updateTree(&embeddedPoints);
        nBodySolvers["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<EmbeddedPoint>(&TSNEbarnesHutReverseMultiPoleParticleNodeKernal, &TSNEbarnesHutReverseMultiPoleParticleParticleKernal, 10, 1.0f);
        nBodySolvers["BHRMP"]->updateTree(&embeddedPoints);
        nBodySolvers["FMM"] = new NBodySolverFMM<EmbeddedPoint>(&TSNEFMMNodeNodeKernal, &TSNEFMMParticleNodeKernal, &TSNEFMMNodeParticleKernal, &TSNEFMMParticleParticleKernal, 10, 1.0f);
        nBodySolvers["FMM"]->updateTree(&embeddedPoints);
        //nBodySolvers["FMMnaive"] = new NBodySolverFMM<EmbeddedPoint>(&TSNEFMMNodeNodeKernalNaive, &TSNEFMMParticleNodeKernalNaive, &TSNEFMMNodeParticleKernalNaive, &TSNEFMMParticleParticleKernal, 10, 1.0f);
        //nBodySolvers["FMMiter"] = new NBodySolverFMMiter<EmbeddedPoint>(&TSNEFMMiterInteractionKernal, 10, 1.0f);
        nBodySolvers["FMMiter"] = new NBodySolverFMMiter<EmbeddedPoint>(&TSNEFMMiterInteractionKernalNodeNode, &TSNEFMMiterInteractionKernalNodeParticle, &TSNEFMMiterInteractionKernalParticleNode, &TSNEFMMiterInteractionKernalParticleParticle, 10, 1.0f);
        nBodySolvers["FMMiter"]->updateTree(&embeddedPoints);

        embeddedBuffer = new Buffer(embeddedPoints, pos2DlabelInt, GL_DYNAMIC_DRAW);

        std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, embeddedDerivative, forceSize);
        forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
	}
	
	~TSNE()
	{
        delete embeddedBuffer;
        delete forceBuffer;

        for (std::pair<const std::string, NBodySolver<EmbeddedPoint>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }
	}

    void cleanup()
    {
        embeddedBuffer->cleanup();
        forceBuffer->cleanup();

        for (std::pair<const std::string, NBodySolver<EmbeddedPoint>*> nBodySolver : nBodySolvers)
        {
            nBodySolver.second->cleanup();
        }
    }
    
    void timeStep()
    {
        timeCounter++;
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            updateDerivative();

            embeddedPointsPrev.swap(embeddedPointsPrevPrev);
            embeddedPoints.swap(embeddedPointsPrev);

            float accelerationRate = 0.8f;
            if (globalTimeStep < 250)
            {
                accelerationRate = 0.5f;
            }

            for (int i = 0; i < embeddedPoints.size(); i++)
            {
                embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddedDerivative[i] + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
            }

            //costFunction();

            embeddedBuffer->updateBuffer(embeddedPoints, pos2DlabelInt);

            nBodySolvers[nBodySelect]->updateTree(&embeddedPoints);

            std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, embeddedDerivative, forceSize);
            forceBuffer->updateBuffer(forceLines, pos2DCol3D);

            globalTimeStep++;
        }
    }

    std::tuple<float, float, float, float> getEdges()
    {
        float left  = std::numeric_limits<float>::max();
        float down  = std::numeric_limits<float>::max();
        float right = std::numeric_limits<float>::min();
        float up    = std::numeric_limits<float>::min();

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            if (embeddedPoints[i].position.x < left) { left = embeddedPoints[i].position.x; }
            if (embeddedPoints[i].position.x > right) { right = embeddedPoints[i].position.x; }
            if (embeddedPoints[i].position.y < down) { down = embeddedPoints[i].position.y; }
            if (embeddedPoints[i].position.y > up) { up = embeddedPoints[i].position.y; }
        }

        return std::make_tuple(left, right, down, up);
    }

private:

    void updateDerivative()
    {
        //checkError();
        
        updateRepulsive();
        
        updateAttractive();

        std::fill(embeddedDerivative.begin(), embeddedDerivative.end(), glm::vec2(0.0f, 0.0f));
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedDerivative[i] = attractForce[i] - repulsForce[i];
        }

    }

    void checkError()
    {
        float QijTotalNaive = 0.0f;
        
        nBodySolvers["naive"]->solveNbody(&QijTotalNaive, &errorCompare, &embeddedPoints);


        float QijTotalCompare = 0.0f;
        nBodySolvers["BH"]->solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints);
        float error1 = 0.0f;
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            error1 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
        }
        error1 /= embeddedPoints.size();
        totalError1 += error1;
        if (error1 > maxError1) { maxError1 = error1; }

        QijTotalCompare = 0.0f;
        nBodySolvers["FMM"]->theta = 1.0f;
        nBodySolvers["FMM"]->solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints);
        float error2 = 0.0f;
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            error2 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
        }
        error2 /= embeddedPoints.size();
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

    void updateRepulsive()
    {
        float QijTotal = 0.0f;

        nBodySolvers[nBodySelect]->solveNbody(&QijTotal, &repulsForce, &embeddedPoints);

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            repulsForce[i] *= (1.0f / QijTotal);
        }
    }

    void updateAttractive()
    {
        std::fill(attractForce.begin(), attractForce.end(), glm::vec2(0.0f, 0.0f));

        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        { 
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it) 
            {
                glm::vec2 diff = embeddedPoints[it.col()].position - embeddedPoints[it.row()].position;
                float distance = glm::length(diff);

                float exageration = 1.0f;
                if (globalTimeStep < 250)
                {
                    exageration = 4.0f;
                }


                attractForce[it.col()] += -exageration * (float)it.value() * (diff / (1.0f + (distance * distance)));
            }
        }
    }

    void costFunction()
    {
        float QijTotal = 0.0f;
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                glm::vec2 diff = embeddedPoints[j].position - embeddedPoints[i].position;
                float distance = glm::length(diff);
                QijTotal += 1.0f / (1.0f + (distance * distance));
                //QijTotal += 1.0f + distance * distance;
            }
        }

        float totalCost = 0.0f;
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                if (it.col() != it.row())
                {
                    glm::vec2 diff = embeddedPoints[it.col()].position - embeddedPoints[it.row()].position;
                    float distance = glm::length(diff);
                    float Qij = (1.0f / (1.0f + (distance * distance))) / QijTotal;
                    //float Qij = 1.0f / (QijTotal / (1.0f + distance * distance));

                    float Pij = (float)it.value();

                    totalCost += Pij * std::log2(Pij / Qij);
                }
            }
        }

        std::cout << "total cost: " << totalCost << std::endl;
    }

};