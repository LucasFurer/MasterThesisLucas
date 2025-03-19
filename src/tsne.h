#ifndef TSNE_H
#define TSNE_H

#include "common.h"
#include "buffer.h"
#include "loader.h"
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>
#include <filesystem>
#include <format>
#include <math.h>
#include <numbers>
#include "nbodysolvers/nBodySolverNaive.h"
#include "nbodysolvers/nBodySolverBarnesHut.h"
#include "nbodysolvers/nBodySolverMultiPole.h"
#include "nbodysolvers/nBodySolverFMM.h"
#include <filesystem>
#include <iostream>
#include <random>

class TSNE
{
public:
    std::vector<EmbeddedPoint> embeddedPoints;
    std::vector<EmbeddedPoint> embeddedPointsPrev;
    std::vector<EmbeddedPoint> embeddedPointsPrevPrev;
    Buffer* embeddedBuffer;

    std::vector<glm::vec2> embeddedDerivative;
    std::vector<glm::vec2> attractForce;
    std::vector<glm::vec2> repulsForce;

    std::vector<glm::vec2> errorCompare;

    NBodySolverNaive<EmbeddedPoint> nBodySolverNaive;
    NBodySolverBarnesHut<EmbeddedPoint> nBodySolverBarnesHut;
    NBodySolverMultiPole<EmbeddedPoint> nBodySolverMultiPole;
    NBodySolverFMM nBodySolverFMM;

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec;
    float lastTimeUpdated;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    //std::vector<std::vector<float>> Qmatrix;
    float Qsum;

    int follow = 1;
    
	TSNE()
	{
        //srand(time(NULL));
        int dataAmount = 10000;
        float perplexity = 30.0f;

        learnRate = 1000.0f;
        accelerationRate = 0.5f;

        timeStepsPerSec = 100000.0f;

        lastTimeUpdated = 0.0f;

        
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

        //Qmatrix.resize(dataAmount);
        //for (int i = 0; i < dataAmount; i++) { Qmatrix[i].resize(dataAmount); }

        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);

        embeddedDerivative.resize(dataAmount);
        attractForce.resize(dataAmount);
        repulsForce.resize(dataAmount);

        errorCompare.resize(dataAmount);


        nBodySolverNaive = NBodySolverNaive<EmbeddedPoint>(&TSNEnaiveKernal);
        nBodySolverBarnesHut = NBodySolverBarnesHut<EmbeddedPoint>(&TSNEbarnesHutParticleNodeKernal, &TSNEbarnesHutParticleParticleKernal);
        nBodySolverMultiPole = NBodySolverMultiPole<EmbeddedPoint>(&TSNEmultiPoleParticleNodeKernal, &TSNEmultiPoleParticleParticleKernal);


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

        embeddedBuffer = new Buffer(embeddedPoints.data(), embeddedPoints.size(), pos2DlabelInt, GL_DYNAMIC_DRAW);
	}
	
	~TSNE()
	{
        // delete embeddedBuffer?
	}

    void cleanup()
    {
        embeddedBuffer->cleanup();
    }
    
    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            updateDerivativeNaive();

            embeddedPointsPrev.swap(embeddedPointsPrevPrev);
            embeddedPoints.swap(embeddedPointsPrev);

            for (int i = 0; i < embeddedPoints.size(); i++)
            {
                embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddedDerivative[i] + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
            }

            embeddedBuffer->updateBufferNew(embeddedPoints.data(), embeddedPoints.size(), pos2DlabelInt);
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

    void updateDerivativeNaive()
    {
        /*
        updateQ();
        std::cout << "new cost is: " << kullbackLeiblerdivergence() << std::endl;

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)
                {
                    glm::vec2 diff = embeddedPoints[j].position - embeddedPoints[i].position;
                    embeddedDerivative[i] += (((float)Pmatrix.coeff(i, j) - Qmatrix[i][j]) * diff) / (1.0f + glm::length(diff));
                    //embeddedDerivative[i] += ((-Qmatrix[i][j]) * diff) / (1.0f + glm::length(diff));
                    //embeddedDerivative[i] += (((float)Pmatrix.coeff(i, j)) * diff) / (1.0f + glm::length(diff));
                }
            }
        }
        */

        std::tuple errorResult = checkError();
        std::cout << "difference in error: " << std::get<0>(errorResult) << std::endl;
        std::cout << "ratio of error:      " << std::get<1>(errorResult) << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        
        updateRepulsive();
        
        updateAttractive();

        std::fill(embeddedDerivative.begin(), embeddedDerivative.end(), glm::vec2(0.0f, 0.0f));
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedDerivative[i] = attractForce[i] - repulsForce[i];
        }
    }

    std::tuple<float, float> checkError()
    {
        float QijTotalNaive = 0.0f;
        
        nBodySolverNaive.solveNbody(&QijTotalNaive, &errorCompare, &embeddedPoints);

        //NBodySolverNaive::solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints);
        //nBodySolverBarnesHut.solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints, 10, 1.0f);
        //nBodySolverMultiPole.solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints, 10, 1.0f);

        //for (int i = 0; i < embeddedPoints.size(); i++)
        //{
        //    errorCompare[i] *= (1.0f / QijTotal);
        //}

        //for (int i = 0; i < embeddedPoints.size(); i++)
        //{
        //    errorCompare[i] = attractForce[i] - repulsForce[i];
        //}

        //-----------------------------------------------------------------------------------

        float QijTotalCompare = 0.0f;
        nBodySolverBarnesHut.solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints, 10, 1.0f);
        float error1 = 0.0f;
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            error1 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
        }
        error1 /= embeddedPoints.size();

        QijTotalCompare = 0.0f;
        nBodySolverMultiPole.solveNbody(&QijTotalCompare, &repulsForce, &embeddedPoints, 10, 1.0f);
        float error2 = 0.0f;
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            error2 += powf(glm::length(repulsForce[i] - errorCompare[i]), 2.0f);
        }
        error2 /= embeddedPoints.size();

        return std::make_tuple(error1 - error2, error1 / error2);
    }

    void updateRepulsive()
    {
        float QijTotal = 0.0f;

        //nBodySolverNaive.solveNbody(&QijTotal, &repulsForce, &embeddedPoints);
        
        //nBodySolverBarnesHut.solveNbody(&QijTotal, &repulsForce, &embeddedPoints, 10, 0.9f); // keep theta between 0.0 (off) and 1.0 (can be higher) 0.3 gives no artifacts

        nBodySolverMultiPole.solveNbody(&QijTotal, &repulsForce, &embeddedPoints, 10, 1.0f);

        //nBodySolverFMM.solveNbody(&QijTotal, &repulsForce, &embeddedPoints, 10, 0.4f);

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            repulsForce[i] *= (1.0f / QijTotal);
        }
    }

    void updateAttractive()
    {
        std::fill(attractForce.begin(), attractForce.end(), glm::vec2(0.0f, 0.0f));

        for (int k = 0; k < Pmatrix.outerSize(); ++k) { // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it) {
                //glm::vec2 diff = embeddedPoints[it.row()].position - embeddedPoints[it.col()].position;
                glm::vec2 diff = embeddedPoints[it.col()].position - embeddedPoints[it.row()].position;
                float distance = glm::length(diff);

                //attractForce[it.col()] += (float)it.value() * (diff / (1.0f + distance));
                attractForce[it.col()] += -(float)it.value() * (diff / (1.0f + distance));
            }
        }
        /*
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)//might be useless
                {
                    glm::vec2 diff = embeddedPoints[j].position - embeddedPoints[i].position;
                    float distance = glm::length(diff);

                    attractForce[i] += (float)Pmatrix.coeff(i, j) * (diff / (1.0f + distance));
                }
            }
        }
        */
    }

    /*
    void updateQ()
    {
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                Qmatrix[i][j] = 0.0f;
            }
        }
        Qsum = 0.0f;


        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)
                {
                    float distance = glm::length(embeddedPoints[i].position - embeddedPoints[j].position);
                    Qmatrix[i][j] = 1.0f / (1.0f + distance);
                    Qsum += Qmatrix[i][j];
                }
            }
        }

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)
                {
                    Qmatrix[i][j] = Qmatrix[i][j] / Qsum;
                }
            }
        }
    }

    float kullbackLeiblerdivergence()
    {
        float cost = 0.0f;

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)
                {
                    if (Qmatrix[i][j] != 0.0f && Pmatrix.coeff(i, j) != 0) 
                    { 
                        //std::cout << "divide by zero" << std::endl; 
                        //std::cout << "put in log: " << (float)Pmatrix.coeff(i, j) / Qmatrix[i][j] << std::endl;
                        cost += (float)Pmatrix.coeff(i, j) * std::log((float)Pmatrix.coeff(i, j) / Qmatrix[i][j]);
                    }

                }
            }
        }

        return cost;
    }
    */

};



#endif