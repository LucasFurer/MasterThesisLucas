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

        timeStepsPerSec = 1000.0f;
        lastTimeUpdated = 0.0f;

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

            embeddedPoints.swap(embeddedPointsPrev);
            embeddedPointsPrev.swap(embeddedPointsPrevPrev);

            for (int i = 0; i < embeddedPoints.size(); i++)
            {
                embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddeDerivative[i] + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrev[i].position);
            }

            embeddedBuffer->updateBufferNew(embeddedPoints.data(), embeddedPoints.size(), pos2DlabelInt);
        }
    }

private:

    void updateDerivativeNaive()
    {
        updateRepulsive();
        
        updateAttractive();

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            //embeddeDerivative[i] = -0.01f * attractForce[i] + -0.0000001f * repulsForce[i];
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
                    repulsForce[i] += result.x;
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
    }
};

#endif