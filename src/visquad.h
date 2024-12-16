#ifndef VISQUAD_H
#define VISQUAD_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class VisQuad
{
public:
    //MoldAgent* moldAgents;
    //std::size_t moldAgentsSize;
    Texture* texture;

    //glm::ivec3* massDensity;
    //std::size_t massDensitySize;
    glm::vec3* colorArr;
    float* densityArr;
    glm::vec3* tempColorArr;
    //float evaporationSpeed;
    //float diffusionSpeed;
    float t;
    float lastTimeSimulated;
    int simAmountPerSec;

    float deltaX;
    float deltaY;

    VisQuad()
    {

    }

    // constructor reads and builds the shader
    VisQuad(int texWidth, int texHeight, float initT, int initSimAmountPerSec, float initDeltaX, float initDeltaY)
    {
        t = initT;
        simAmountPerSec = initSimAmountPerSec;
        lastTimeSimulated = 0.0f;

        deltaX = initDeltaX;
        deltaY = initDeltaY;


        texture = new Texture(texWidth, texHeight);

        colorArr = new glm::vec3[texWidth * texHeight];
        densityArr = new float[texWidth * texHeight];
        for (int i = 0; i < texWidth * texHeight; i++)
        {
            colorArr[i] = glm::vec3(0.0f);
            densityArr[i] = float(0.0f);
        }
        tempColorArr = new glm::vec3[texWidth * texHeight];




        //massDensity = new glm::ivec3[1];
        //massDensitySize = 1 * sizeof(glm::ivec3);
        int halfX = texWidth / 2;
        int halfY = texHeight / 2;
        int thirdX = texWidth / 3;
        int thirdY = texHeight / 3;
        //massDensity[0] = glm::ivec3(halfX, halfY, 1);
        //colorArr[halfX + halfY * texWidth] = glm::vec3(1.0f);
        densityArr[halfX + halfY * texWidth] = 1.0f;
        densityArr[thirdX + thirdY * texWidth] = 1.0f;
        densityArr[2 * thirdX + halfY * texWidth] = 1.0f;


        //int thirdX = texWidth / 3;
        //int thirdY = texHeight / 3;
        //massDensity[1] = glm::ivec3(thirdX, thirdY, 1);
        //massDensity[2] = glm::ivec3(2 * thirdX, thirdY, 1);
    }


    void updateVisual()
    {
        float currentTime = glfwGetTime();
        if (currentTime - lastTimeSimulated > 1.0f / (float)simAmountPerSec)
        {
            lastTimeSimulated = currentTime;

            //for (int i = 0; i < massDensitySize / sizeof(glm::ivec3); i++)
            //{
            //    int index = massDensity[i].x + massDensity[i].y * texture->width;
            //    //colorArr[index] = glm::vec3((float)massDensity[i].z / 1000.0f);
            //}

            for (int y = 1; y < texture->height - 1; y++)
            {
                for (int x = 1; x < texture->width - 1; x++)
                {
                    glm::vec3 sum(0.0f);
                    tempColorArr[x + y * texture->width] = 0.25f * (
                        colorArr[(x - 1) + y * texture->width] +
                        colorArr[(x + 1) + y * texture->width] +
                        colorArr[x + (y - 1) * texture->width] +
                        colorArr[x + (y + 1) * texture->width] +
                        densityArr[x + y * texture->width]
                        );
                }
            }
            /*
            for (int i = 0; i < texture->width * texture->height; i++)
            {
                int Xcoord = i % texture->width;
                int Ycoord = i / texture->width;

                glm::vec3 sum(0.0f);

                if (Xcoord - 1 >= 0) { sum += colorArr[(Xcoord - 1) + (Ycoord * texture->width)] / powf(deltaX, 2.0f); }
                if (Xcoord + 1 < texture->width) { sum += colorArr[(Xcoord + 1) + (Ycoord * texture->width)] / powf(deltaX, 2.0f); }
                if (Ycoord - 1 >= 0) { sum += colorArr[Xcoord + ((Ycoord - 1) * texture->width)] / powf(deltaY, 2.0f); }
                if (Ycoord + 1 < texture->height) { sum += colorArr[Xcoord + ((Ycoord + 1) * texture->width)] / powf(deltaY, 2.0f); }
                sum -= 2.0f * colorArr[Xcoord + (Ycoord * texture->width)] / powf(deltaX, 2.0f);
                sum -= 2.0f * colorArr[Xcoord + (Ycoord * texture->width)] / powf(deltaY, 2.0f);

                tempColorArr[i] = sum;
            }



            */

            /*
            for (int y = 1; y < texture->height-1; y++)
            {
                for (int x = 1; x < texture->width-1; x++)
                {
                    std::cout << tempColorArr[x + y * texture->width].x << ", ";
                }
                std::cout << std::endl;
            }
            */

            for (int y = 1; y < texture->height - 1; y++)
            {
                for (int x = 1; x < texture->width - 1; x++)
                {
                    /*
                    int x = i % texture->width;
                    int y = i / texture->width;
                    int devide = 0;
                    colorArr[i] = glm::vec3(0.0f);
                    if (x - 1 >= 0)              { colorArr[i] += tempColorArr[(x - 1) +  y      * texture->width]; devide++; }
                    if (x + 1 < texture->width)  { colorArr[i] += tempColorArr[(x + 1) +  y      * texture->width]; devide++; }
                    if (y - 1 >= 0)              { colorArr[i] += tempColorArr[ x      + (y - 1) * texture->width]; devide++; }
                    if (y + 1 < texture->height) { colorArr[i] += tempColorArr[ x      + (y + 1) * texture->width]; devide++; }
                    //colorArr[i] = tempColorArr[i];
                    colorArr[i] /= devide * 7.5f;
                    */
                    int i = x + y * texture->width;

                    colorArr[i] = tempColorArr[i];
                    /*
                    texture->data[4 * i + 0] = 255 * ((2.0f / (1.0f + powf(E, colorArr[i].r))) - 1.0f);
                    texture->data[4 * i + 1] = 255 * ((2.0f / (1.0f + powf(E, colorArr[i].g))) - 1.0f);
                    texture->data[4 * i + 2] = 255 * ((2.0f / (1.0f + powf(E, colorArr[i].b))) - 1.0f);
                    */
                    texture->data[4 * i + 0] = 255 * colorArr[i].r;
                    texture->data[4 * i + 1] = 255 * colorArr[i].g;
                    texture->data[4 * i + 2] = 255 * colorArr[i].b;
                    texture->data[4 * i + 3] = 255;
                }
            }

            texture->updateData();
        }
    }

    ~VisQuad()
    {
        //delete[] massDensity;
        delete[] colorArr;
        delete[] densityArr;
        delete[] tempColorArr;
    }

private:
};


#endif