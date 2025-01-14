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

    std::vector<glm::vec2> embeddeDerivative;
    std::vector<glm::vec2> attractForce;
    std::vector<glm::vec2> repulsForce;

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec;
    float lastTimeUpdated;

    Eigen::SparseMatrix<double> Pmatrix;
    
	TSNE()
	{
        //srand(time(NULL));
        int dataAmount = 1000;
        float perplexity = 30.0f;

        learnRate = 1.0f;
        accelerationRate = 0.0f;

        timeStepsPerSec = 60.0f;
        lastTimeUpdated = 0.0f;


        std::string fileName = "data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx";
        //std::string fileName = "data/P_matrix_amount1000_perp30.mtx";
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
        dataAmount = 100;

        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);

        embeddeDerivative.resize(dataAmount);
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

            int lab = 0;
            
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
                //embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddeDerivative[i] + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
                embeddedPoints[i].position = embeddedPointsPrev[i].position + glm::vec2(0.01f);
            }

            std::cout << "print all" << std::endl;
            for (int i = 0; i < embeddedPoints.size(); i++)
            {
                std::cout << glm::to_string(embeddedPoints[i].position) << std::endl;
            }
            std::cout << "print done" << std::endl;

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
        updateRepulsive();
        
        updateAttractive();

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddeDerivative[i] = -0.01f * attractForce[i] + -0.000001f * repulsForce[i];
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
                if (i != j)
                {
                    glm::vec2 iminj = embeddedPoints[i].position - embeddedPoints[j].position;
                    float distance = iminj.length();

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

    void updateAttractive()
    {
        std::fill(attractForce.begin(), attractForce.end(), glm::vec2(0.0f, 0.0f));

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            for (int j = 0; j < embeddedPoints.size(); j++)
            {
                if (i != j)
                {
                    glm::vec2 iminj = embeddedPoints[i].position - embeddedPoints[j].position;
                    float distance = iminj.length();

                    //qijTotal += (1.0f + distance);

                    glm::vec2 result = iminj / (1.0f + distance);
                    //attractForce[i] += (float)Pmatrix.coeff(i, j) * result;
                }
            }
        }
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            //attractForce[i] *= 4.0f;
        }
    }
};

#endif