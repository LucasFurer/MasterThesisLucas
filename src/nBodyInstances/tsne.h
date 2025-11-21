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
#include "../particles/tsnePoint2D.h"
#include "../openGLhelper/buffer.h"
#include "../common.h"
#include "../openGLhelper/buffer.h"
#include "../dataLoaders/loader.h"
#include "../nbodysolvers/cpu/nBodySolverNaive.h"
#include "../nbodysolvers/cpu/nBodySolverBH.h"
#include "../nbodysolvers/cpu/nBodySolverBHR.h"
#include "../nbodysolvers/cpu/nBodySolverBarnesHutReverseMultiPole.h"
#include "../nbodysolvers/cpu/nBodySolverBHMP.h"
#include "../nbodysolvers/cpu/nBodySolverFMM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"


class TSNE
{
public:
    std::vector<TsnePoint2D> embeddedPoints;
    std::vector<TsnePoint2D> embeddedPointsPrev;
    std::vector<TsnePoint2D> embeddedPointsPrevPrev;
    Buffer* embeddedBuffer;

    int nodeLevelToShow = 0;
    Buffer* nodeBuffer;

    float forceSize = 1.0f;
    Buffer* forceBuffer;

    //std::vector<glm::vec2> embeddedDerivative;
    //std::vector<glm::vec2> attractForce;
    //std::vector<glm::vec2> repulsForce;

    //std::vector<glm::vec2> errorCompare;

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect = "naive";

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec;
    float lastTimeUpdated;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    //std::vector<std::vector<float>> Qmatrix;
    //float Qsum;

    int follow = 1;
    //float totalError = 0.0f;
    int timeCounter = 0;

    //float totalError1 = 0.0f;
    //float totalError2 = 0.0f;
    //float maxError1 = 0.0f;
    //float maxError2 = 0.0f;
    int globalTimeStep = 0;
    
	TSNE()
	{
        int dataAmount = 1000;
        float perplexity = 30.0f;
        std::string dataSet = "MNIST_digits";
        //std::string dataSet = "CIFAR10";

        learnRate = 200.0f;
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

        //embeddedDerivative.resize(dataAmount);
        //attractForce.resize(dataAmount);
        //repulsForce.resize(dataAmount);

        //errorCompare.resize(dataAmount);

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
            
            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
        }

        nBodySolvers["naive"] = new NBodySolverNaive<TsnePoint2D>(&TSNEnaiveKernel);
        nBodySolvers["BH"] = new NBodySolverBH<TsnePoint2D>(&TSNEBHPNKernel, &TSNEBHPPKernel, 10, 1.0f);
        nBodySolvers["BH"]->updateTree(embeddedPoints);
        nBodySolvers["BHR"] = new NBodySolverBHR<TsnePoint2D>(&TSNEBHRNPKernel, &TSNEBHRPPKernel, 10, 1.0f);
        nBodySolvers["BHR"]->updateTree(embeddedPoints);
        nBodySolvers["BHMP"] = new NBodySolverBHMP<TsnePoint2D>(&TSNEBHMPPNKernel, &TSNEBHMPPPKernel, 10, 1.0f);
        nBodySolvers["BHMP"]->updateTree(embeddedPoints);
        //nBodySolvers["BHRMP"] = new NBodySolverBarnesHutReverseMultiPole<TsnePoint2D>(&TSNEbarnesHutReverseMultiPoleParticleNodeKernel, &TSNEbarnesHutReverseMultiPoleParticleParticleKernel, 10, 1.0f);
        //nBodySolvers["BHRMP"]->updateTree(&embeddedPoints);
        //nBodySolvers["FMM"] = new NBodySolverFMM<TsnePoint2D>(&TSNEFMMNodeNodeKernel, &TSNEFMMParticleNodeKernel, &TSNEFMMNodeParticleKernel, &TSNEFMMParticleParticleKernel, 10, 1.0f);
        //nBodySolvers["FMM"]->updateTree(&embeddedPoints);
        
        //nBodySolvers["FMMnaive"] = new NBodySolverFMM<TsnePoint2D>(&TSNEFMMNodeNodeKernelNaive, &TSNEFMMParticleNodeKernelNaive, &TSNEFMMNodeParticleKernelNaive, &TSNEFMMParticleParticleKernel, 10, 1.0f);
        //nBodySolvers["FMMiter"] = new NBodySolverFMMiter<TsnePoint2D>(&TSNEFMMiterInteractionKernel, 10, 1.0f);
        //nBodySolvers["FMMiter"] = new NBodySolverFMMiter<TsnePoint2D>(&TSNEFMMiterInteractionKernelNodeNode, &TSNEFMMiterInteractionKernelNodeParticle, &TSNEFMMiterInteractionKernelParticleNode, &TSNEFMMiterInteractionKernelParticleParticle, 10, 1.0f);
        //nBodySolvers["FMMiter"]->updateTree(&embeddedPoints);

        embeddedBuffer = new Buffer(embeddedPoints, Float2Float2Int1Int1, GL_DYNAMIC_DRAW);

        std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
        nodeBuffer = new Buffer(nodesBufferData, pos2DCol3D, GL_DYNAMIC_DRAW);
        
        //std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, embeddedDerivative, forceSize);
        //forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
	}
	
	~TSNE()
	{
        delete embeddedBuffer;
        delete nodeBuffer;
        //delete forceBuffer;

        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }
	}

    void cleanup()
    {
        embeddedBuffer->cleanup();
        nodeBuffer->cleanup();
        //forceBuffer->cleanup();

        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolver : nBodySolvers)
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
                embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddedPointsPrev[i].derivative + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
                embeddedPoints[i].derivative = embeddedPointsPrev[i].derivative; // for showing the derivatives
            }

            //costFunction();

            embeddedBuffer->updateBuffer(embeddedPoints, Float2Float2Int1Int1);

            nBodySolvers[nBodySelect]->updateTree(embeddedPoints);
            std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
            nodeBuffer->updateBuffer(nodesBufferData, pos2DCol3D);

            //std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, embeddedDerivative, forceSize);
            //forceBuffer->updateBuffer(forceLines, pos2DCol3D);

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
        for (TsnePoint2D embeddedPoint : embeddedPoints)
            embeddedPoint.derivative = glm::vec2(0.0f); // reset derivative for iteration

        //checkError();
        
        updateRepulsive();
        
        updateAttractive();
    }

    //void checkError()
    //{
    //    float QijTotalNaive = 0.0f;
    //    
    //    nBodySolvers["naive"]->solveNbody(&QijTotalNaive, &errorCompare, &embeddedPoints);
    //
    //
    //    float QijTotalCompare = 0.0f;
    //    nBodySolvers["BH"]->solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints);
    //    float error1 = 0.0f;
    //    for (int i = 0; i < embeddedPoints.size(); i++)
    //    {
    //        error1 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
    //    }
    //    error1 /= embeddedPoints.size();
    //    totalError1 += error1;
    //    if (error1 > maxError1) { maxError1 = error1; }
    //
    //    QijTotalCompare = 0.0f;
    //    nBodySolvers["FMM"]->theta = 1.0f;
    //    nBodySolvers["FMM"]->solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints);
    //    float error2 = 0.0f;
    //    for (int i = 0; i < embeddedPoints.size(); i++)
    //    {
    //        error2 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
    //    }
    //    error2 /= embeddedPoints.size();
    //    totalError2 += error2;
    //    if (error2 > maxError2) { maxError2 = error2; }
    //
    //
    //
    //    std::cout << "difference in error:         " << error1 - error2 << " ,greater than 0.0 is good" << std::endl;
    //    std::cout << "average error difference:    " << (totalError1 / globalTimeStep) - (totalError2 / globalTimeStep) << " ,greater than 0.0 is good" << std::endl;
    //
    //    std::cout << "max error1:                  " << maxError1 << std::endl;
    //    std::cout << "max error2:                  " << maxError2 << std::endl;
    //
    //    std::cout << "ratio of error:              " << error1 / error2 << " ,greater than 1.0 is good" << std::endl;
    //    std::cout << "ratio of average error:      " << (totalError1 / globalTimeStep) / (totalError2 / globalTimeStep) << " ,greater than 1.0 is good" << std::endl;
    //
    //    std::cout << "global time step: " << globalTimeStep << std::endl;
    //
    //    std::cout << "------------------------------------------------------" << std::endl;
    //}

    void updateRepulsive()
    {
        float QijTotal = 0.0f;

        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints);

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedPoints[i].derivative *= (4.0f / QijTotal);
        }
    }

    void updateAttractive()
    {
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        { 
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it) 
            {
                glm::vec2 diff = embeddedPoints[it.row()].position - embeddedPoints[it.col()].position;
                float dist = glm::length(diff);

                float exageration = 1.0f;
                if (globalTimeStep < 250)
                {
                    exageration = 4.0f;
                }

                embeddedPoints[it.col()].derivative += exageration * 4.0f * (float)it.value() * (diff / (1.0f + (dist * dist)));
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