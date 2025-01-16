#ifndef TSNE_H
#define TSNE_H

#include "common.h"
#include "buffer.h"
#include "loader.h"
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Geometry>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>

#include <filesystem>
#include <format>
#include <math.h>
#include <numbers>

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

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec;
    float lastTimeUpdated;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    std::vector<std::vector<float>> Qmatrix;
    float Qsum;
    
	TSNE()
	{
        //srand(time(NULL));
        int dataAmount = 1000;
        float perplexity = 30.0f;

        learnRate = 10.0f;
        accelerationRate = 0.5f;

        timeStepsPerSec = 60.0f;
        lastTimeUpdated = 0.0f;


        std::string labelsPath = "data/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin";
        labels = Loader::loadLabels(labelsPath);
        
        std::string fileName = "data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx";
        std::ifstream file(fileName);
        if (file.is_open())
        {
            Eigen::loadMarket(Pmatrix, fileName);
            std::cout << "Matrix loaded successfully!" << std::endl;
            //Pmatrix.coeff(i, j)
            //Pmatrix.rows()
            //Pmatrix.cols()
        }
        else
        {
            std::cerr << "Failed to open " + fileName + " file!" << std::endl;
        }
        Qmatrix.resize(dataAmount);
        for (int i = 0; i < dataAmount; i++) { Qmatrix[i].resize(dataAmount); }


        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);

        embeddedDerivative.resize(dataAmount);
        attractForce.resize(dataAmount);
        repulsForce.resize(dataAmount);

        float sizeParam = 200.0f;
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
        delete embeddedBuffer;
        //maybe do this delete embeddedBuffer
        //actually why would i even have dynamic buffer?
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
                }
            }
        }



        /*
        updateRepulsive();
        
        updateAttractive();

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            //embeddeDerivative[i] = -0.01f * attractForce[i] + -0.000001f * repulsForce[i];
            embeddeDerivative[i] = 1.0f * attractForce[i] + 1.0f * repulsForce[i];
        }
        */
    }

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

    void updateRepulsive()
    {
        std::fill(repulsForce.begin(), repulsForce.end(), glm::vec2(0.0f, 0.0f));

        float qijTotal = 0.0f;

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)//might be useless
                {
                    glm::vec2 iminj = embeddedPoints[i].position - embeddedPoints[j].position;
                    float distance = glm::length(iminj);

                    qijTotal += (1.0f + distance);

                    glm::vec2 result = iminj / ((1.0f + distance) * (1.0f + distance));
                    repulsForce[i] += result;
                }
            }
        }
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            repulsForce[i] *= -4.0f * qijTotal;
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
};

#endif