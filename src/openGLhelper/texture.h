#pragma once

#include <glad/glad.h>
#include <iostream>

#include "../stb_image.h"

class Texture
{
public:
    unsigned int TEX = 0;
    int width;
    int height;
    int nrChannels;
    unsigned char* data;

	Texture(const char* fileName)
	{
        glGenTextures(1, &TEX);
        glBindTexture(GL_TEXTURE_2D, TEX);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        //int stbiWidth, stbiHeight, stbiNrChannels;
        stbi_set_flip_vertically_on_load(true);
        //unsigned char* data = stbi_load(fileName, &width, &height, &nrChannels, 0);
        data = stbi_load(fileName, &width, &height, &nrChannels, 0);

        if (data)
        {
            switch (nrChannels) 
            {
            case 4:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                glGenerateMipmap(GL_TEXTURE_2D);
                break;
            case 3:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
                glGenerateMipmap(GL_TEXTURE_2D);
                break;
            default:
                std::cout << "unsupported number of channels for texture" << std::endl;
            }

        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
        }
        //stbi_image_free(data);
        //glBindTexture(GL_TEXTURE_2D, 0);
	}

    Texture(int initWidth, int initHeight)
    {
        width = initWidth;
        height = initHeight;
        nrChannels = 4;

        glGenTextures(1, &TEX);
        glBindTexture(GL_TEXTURE_2D, TEX);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        data = new unsigned char[initWidth * initHeight * 4];
        for (int i = 0; i < initWidth * initHeight * 4; i++)
        {
            data[i] = (unsigned char)0;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, initWidth, initHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        //glGenerateMipmap(GL_TEXTURE_2D);
    }


	~Texture()
	{
        stbi_image_free(data);

        if (TEX != 0) { std::cerr << "VAO of buffer was not deleted!" << std::endl; }
	}

    void cleanup()
    {
        if (TEX != 0) { glDeleteTextures(1, &TEX); TEX = 0; }
    }

    void updateData()
    {
        /*
        std::cout << width << std::endl;
        std::cout << height << std::endl;
        std::cout << nrChannels << std::endl;
        std::cout << (int)data[0] << std::endl;
        std::cout << (int)data[1] << std::endl;
        std::cout << (int)data[2] << std::endl;
        std::cout << (int)data[3] << std::endl;
        */

        glBindTexture(GL_TEXTURE_2D, TEX);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
private:

};