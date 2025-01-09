#ifndef TSNE_H
#define TSNE_H

#include "common.h"
//#include <vector>
#include "buffer.h"
#include "loader.h"

class TSNE
{
public:
    float* dataP;
    unsigned int dataPAmount;
    unsigned int dataPDimension;
    int* labelsP;

	Particle2D* dataQ;
    Particle2D* dataQPrev;
    Particle2D* dataQPrevPrev;
	std::size_t dataQSize;
	Buffer* dataQBuffer;

    float* attractForce;
    float* repulsForce;
    float* dataQDerivative;

    float learnRate;
    float accelerationRate;
    float perplexity;
    float* sigma;

    float timeStepsPerSec;
    float lastTimeUpdated;
    
	TSNE()
	{
        //srand(time(NULL));
        learnRate = 1.0f;
        accelerationRate = 0.0f;
        perplexity = 4.0f;

        timeStepsPerSec = 1000.0f;
        lastTimeUpdated = 0.0f;

        loadData1();
        std::filesystem::path currentPath = std::filesystem::current_path();
        std::filesystem::path newPath = currentPath / "data\\t10k-images.idx3-ubyte";
        loadData2(newPath.string().c_str());

        sigma = new float[dataPAmount];
        std::fill(sigma, sigma + dataPAmount, 1.0f);


        dataQ = new Particle2D[dataPAmount];
        dataQPrev = new Particle2D[dataPAmount];
        dataQPrevPrev = new Particle2D[dataPAmount];
        dataQSize = dataPAmount * sizeof(Particle2D);

        dataQDerivative = new float[dataPAmount * 2];
        memset(dataQDerivative, 0, sizeof(float) * dataPAmount * 2);
        attractForce = new float[dataPAmount * 2];
        memset(attractForce, 0, sizeof(float) * dataPAmount * 2);
        repulsForce = new float[dataPAmount * 2];
        memset(repulsForce, 0, sizeof(float) * dataPAmount * 2);

        float sizeParam = 20.0f;
        for (int i = 0; i < dataPAmount; i++)
        {
            glm::vec2 pos = glm::vec2(0.0f);
            glm::vec2 speed = glm::vec2(0.0f);
            glm::vec3 col = glm::vec3(0.0f);

            float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            while (powf(randX, 2.0f) + powf(randY, 2.0f) + powf(randZ, 2.0f) > 1.0f)
            {
                randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randZ = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }

            pos = glm::vec2(
                powf(sizeParam * randX, 1.0f),
                powf(sizeParam * randY, 1.0f)
            );
            //col = glm::vec3(((float)rand() / RAND_MAX), ((float)rand() / RAND_MAX), ((float)rand() / RAND_MAX));

            if (i < 4)
            {
                col = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            else if (i < 7)
            {
                col = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            else
            {
                col = glm::vec3(0.0f, 0.0f, 1.0f);
            }

            dataQ[i] = Particle2D(pos, speed, col, 1.0f);
            dataQPrev[i] = Particle2D(pos, speed, col, 1.0f);
            dataQPrevPrev[i] = Particle2D(pos, speed, col, 1.0f);
        }
        float* particlesToBuffer = Particle2D::Particle2DToFloat(dataQ, dataQSize);
        dataQBuffer = new Buffer(particlesToBuffer, 5 * sizeof(float) * (dataQSize / sizeof(Particle2D)), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] particlesToBuffer;

        updateSigma();
	}
	
	~TSNE()
	{
		delete[] dataP;
		delete[] dataQ;
		delete[] dataQPrev;
		delete[] dataQPrevPrev;
        delete[] sigma;

        delete[] attractForce;
        delete[] repulsForce;

        delete[] dataQDerivative;
	}
    
    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            updateDerivativeNaive();

            Particle2D* temp = dataQPrevPrev;   // shift current, prev, and prevprev
            dataQPrevPrev = dataQPrev;
            dataQPrev = dataQ;
            dataQ = temp;



            for (int i = 0; i < dataPAmount; i++)
            {
                dataQ[i].position.x = dataQPrev[i].position.x + learnRate * dataQDerivative[2 * i + 0] + accelerationRate * (dataQPrev[i].position.x - dataQPrevPrev[i].position.x);
                dataQ[i].position.y = dataQPrev[i].position.y + learnRate * dataQDerivative[2 * i + 1] + accelerationRate * (dataQPrev[i].position.y - dataQPrevPrev[i].position.y);
            }

            float* toBuffer = Particle2D::Particle2DToFloat(dataQ, dataQSize);
            dataQBuffer->updateBuffer(toBuffer, 5 * sizeof(float) * (dataQSize / sizeof(Particle2D)), pos2DCol3D);
            delete[] toBuffer;
        }
    }

private:

    void updateDerivativeNaive()
    {
        /*
        std::cout << "all points: " << std::endl;
        for (int i = 0; i < dataPAmount; i++)
        {
            std::cout << glm::to_string(dataQ[i].position) << std::endl;
        }
        */

        updateRepulsive();
        
        updateAttractive();

        memset(dataQDerivative, 0, sizeof(float) * dataPAmount * 2);
        for (int i = 0; i < dataPAmount; i++)
        {
            dataQDerivative[2 * i + 0] = -0.01f * attractForce[2 * i + 0] + -0.0000001f * repulsForce[2 * i + 0];
            dataQDerivative[2 * i + 1] = -0.01f * attractForce[2 * i + 1] + -0.0000001f * repulsForce[2 * i + 1];
        }
    }

    void updateRepulsive()
    {
        memset(repulsForce, 0, sizeof(float) * dataPAmount * 2);

        float qijTotal = 0.0f;

        for (int i = 0; i < dataPAmount; i++)
        {
            for (int j = 0; j < dataPAmount; j++)
            {
                if (i != j)
                {
                    glm::vec2 iminj = dataQ[i].position - dataQ[j].position;
                    float distance = iminj.length();

                    qijTotal += (1.0f + distance);

                    glm::vec2 result = iminj / ((1.0f + distance) * (1.0f + distance));
                    repulsForce[2 * i + 0] += result.x;
                    repulsForce[2 * i + 1] += result.y;
                }
            }
        }
        for (int i = 0; i < dataPAmount; i++)
        {
            repulsForce[2 * i + 0] *= -4.0f * qijTotal;
            repulsForce[2 * i + 1] *= -4.0f * qijTotal;
        }
    }

    void updateAttractive()
    {
        memset(attractForce, 0, sizeof(float) * dataPAmount * 2);

        float* totalDivide = new float[dataPAmount];
        memset(totalDivide, 0, sizeof(float) * dataPAmount);

        for (int i = 0; i < dataPAmount; i++)
        {
            for (int j = 0; j < dataPAmount; j++)
            {
                if (i != j)
                {
                    totalDivide[i] += pow(E, -pDistance(i, j) / (2.0f * sigma[i] * sigma[i]));
                    totalDivide[j] += pow(E, -pDistance(j, i) / (2.0f * sigma[j] * sigma[j]));
                }
            }
        }

        for (int i = 0; i < dataPAmount; i++)
        {
            for (int j = 0; j < dataPAmount; j++)
            {
                if (i != j)
                {
                    float Pij = ((pow(E, -pDistance(i, j) / (2.0f * sigma[i] * sigma[i])) / totalDivide[i]) +
                                 (pow(E, -pDistance(j, i) / (2.0f * sigma[j] * sigma[j])) / totalDivide[j])) / (2.0f * (float)dataPAmount);

                    glm::vec2 diffVec = dataQ[i].position - dataQ[j].position;
                    glm::vec2 force = (Pij / (1.0f + diffVec.length())) * diffVec;
                    attractForce[2 * i + 0] += force.x;
                    attractForce[2 * i + 1] += force.y;
                }
            }
        }

        for (int i = 0; i < dataPAmount; i++)
        {
            attractForce[2 * i + 0] *= 4.0f;
            attractForce[2 * i + 1] *= 4.0f;
        }

        delete[] totalDivide;
    }

    void updateSigma()
    {
        //std::fill(sigma, sigma + dataPAmount, 1.0f);
        
        for (int i = 0; i < dataPAmount; i++)
        {
            int wentUp = 2;
            bool searching = true;
            int halving = 5;
            float lastChange = 0.0f;


            while (searching)
            {
                float calculatedPerplexity = getPerplexity(i);

                if (wentUp == 2)
                {
                    if (calculatedPerplexity > perplexity)
                    {
                        sigma[i] /= 2.0f;
                        lastChange = sigma[i];
                        wentUp = 0;
                    }
                    else
                    {
                        sigma[i] *= 2.0f;
                        lastChange = sigma[i]/2.0f;
                        wentUp = 1;
                    }
                }
                else if (wentUp == 1)
                {
                    if (calculatedPerplexity > perplexity)
                    {
                        sigma[i] /= 2.0f;
                        lastChange = sigma[i];
                        wentUp = 0;
                        searching = false;
                    }
                    else
                    {
                        sigma[i] *= 2.0f;
                        lastChange = sigma[i] / 2.0f;
                        wentUp = 1;
                    }
                }
                else
                {
                    if (calculatedPerplexity > perplexity)
                    {
                        sigma[i] /= 2.0f;
                        lastChange = sigma[i];
                        wentUp = 0;
                    }
                    else
                    {
                        sigma[i] *= 2.0f;
                        lastChange = sigma[i] / 2.0f;
                        wentUp = 1;
                        searching = false;
                    }
                }
            }

            for (int d = 0; d < halving; d++)
            {
                float calculatedPerplexity = getPerplexity(i);

                if (calculatedPerplexity > perplexity)
                {
                    lastChange /= 2.0f;
                    sigma[i] -= lastChange;
                }
                else
                {
                    lastChange /= 2.0f;
                    sigma[i] += lastChange;
                }
            }

        }
        
    }

    float getPerplexity(int i)
    {
        float PDivide = 0.0f;
        for (int k = 0; k < dataPAmount; k++)
        {
            if (i != k)
            {
                PDivide += pow(E, pDistance(i, k) / (2.0f * sigma[i] * sigma[i]));
            }
        }

        float PjiSum = 0.0f;
        for (int j = 0; j < dataPAmount; j++)
        {
            float Pji = pow(E, pDistance(i, j) / (2.0f * sigma[i] * sigma[i])) / PDivide;
            PjiSum -= Pji * std::log2(Pji);
        }

        float calculatedPerplexity = pow(2.0f, PjiSum);
        return calculatedPerplexity;
    }

    float pDistance(int i, int j)
    {
        float distance = 0.0f;
        for (int d = 0; d < dataPDimension; d++)
        {
            distance += pow(dataP[i * dataPDimension + d] - dataP[j * dataPDimension + d], 2.0f);
        }

        return sqrt(distance);
    }

    void loadData1()
    {
        dataPAmount = 10;
        dataPDimension = 3;
        dataP = new float[dataPAmount * dataPDimension];

        dataP[0] = 10.0f;
        dataP[1] = 10.0f;
        dataP[2] = 10.0f;

        dataP[3] = 11.0f;
        dataP[4] = 9.0f;
        dataP[5] = 12.0f;
        
        dataP[6] = 8.6f;
        dataP[7] = 9.2f;
        dataP[8] = 10.4f;

        dataP[9] = 13.2f;
        dataP[10] = 12.1f;
        dataP[11] = 9.2f;

        //-------------

        dataP[12] = -8.2f;
        dataP[13] = 11.3f;
        dataP[14] = 14.2f;

        dataP[15] = -11.0f;
        dataP[16] = 13.0f;
        dataP[17] = 8.0f;

        dataP[18] = -7.0f;
        dataP[19] = 14.0f;
        dataP[20] = 12.0f;

        //----------------------------------

        dataP[21] = 8.0f;
        dataP[22] = -9.0f;
        dataP[23] = -9.0f;

        dataP[24] = 12.0f;
        dataP[25] = -11.0f;
        dataP[26] = -9.0f;

        dataP[27] = 12.0f;
        dataP[28] = -13.0f;
        dataP[29] = -11.0f;
        
    }

    void loadData2(const char* path)
    {
        Loader::loadMNIST(dataP, &dataPAmount, &dataPDimension, path);

        std::cout << dataPAmount << std::endl;
        std::cout << dataPDimension << std::endl;
        std::cout << dataP[0] << std::endl;
    }

};

#endif