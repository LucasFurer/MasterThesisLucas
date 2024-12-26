#ifndef TSNE_H
#define TSNE_H

//#include "octtree.h"
//#include "ffthelper.h"
//#include "common.h"
//#include <vector>
#include "buffer.h"

class TSNE
{
public:
    float* dataP;
    unsigned int dataPAmount;
    unsigned int dataPDimension;

	Particle2D* dataQ;
	std::size_t dataQSize;
	Buffer* dataQBuffer;


	//TSNE()
	//{
            
	//}
    
	TSNE()
	{
        loadCustomData();

        dataQ = new Particle2D[dataPAmount];
        dataQSize = dataPAmount * sizeof(Particle2D);

        float sizeParam = 30.0f;
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

            speed = glm::vec2(
                (((float)rand() / RAND_MAX) - 0.5f) / 30.0f,
                (((float)rand() / RAND_MAX) - 0.5f) / 30.0f
            );

            col = glm::vec3(1.0f);
            
            //col = glm::vec3(
            //    randX * 0.5f + 0.5f,
            //    randY * 0.5f + 0.5f,
            //    randZ * 0.5f + 0.5f
            //);
            
            dataQ[i] = Particle2D(pos, speed, col, 1.0f);
        }
        float* particlesToBuffer = Particle2D::Particle2DToFloat(dataQ, dataQSize);
        dataQBuffer = new Buffer(particlesToBuffer, 5 * sizeof(float) * (dataQSize / sizeof(Particle2D)), pos2DCol3D, GL_DYNAMIC_DRAW);
        delete[] particlesToBuffer;
	}
	
	~TSNE()
	{
		delete[] dataP;
		delete[] dataQ;
	}
    
    void timeStep()
    {

    }

    void loadCustomData()
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
    
private:

};

#endif