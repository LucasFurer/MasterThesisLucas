#pragma once
// this whole class is a mess, clean up!!!!!!!!!!

#include <glad/glad.h>
#include <vector>
#include <iostream>
#include <cstddef>

enum DataType
{
	pos3DNorm3DCol3DUV2D,
	pos3D,
	pos3DCol3D,
	pos3DUV2D,
	pos2DCol3D,
	pos2DlabelInt,
	pos2Dvel2Dcol3Dmass,
	pos3DNOTvel3DCol3DNOTmass1D,
	pos2Dcol3Dpos2Dcol3DNOTdepth1D,
	Pos2FloatAcc2FloatLab1Int,
	Float2Float2Int1Int1
};

class Buffer
{
public:
	unsigned int VBO = 0;
	unsigned int VAO = 0;
	unsigned int EBO = 0;
	int elementAmount;

	// constructor --------------------------------------------------------------------------------------------------------

	Buffer() // maybe dont create buffers on an empty buffer object because who is going to clean this up
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);
		elementAmount = 0;
	}

	template <typename T>
	Buffer(const std::vector<T>& toBuffer, std::vector<unsigned int> indices, DataType dataType, GLenum bufferType) // deprecated
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);
		//elementAmount = indices.size();

		createElementBuffer(toBuffer, indices, dataType, bufferType);
	}

	template <typename T>
	Buffer(const std::vector<T>& toBuffer, DataType dataType, GLenum bufferType)
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		//elementAmount = toBuffer.size();

		createVertexBuffer(toBuffer, dataType, bufferType);
	}

	~Buffer()
	{
		if (VAO != 0) { std::cerr << "VAO of buffer was not deleted!" << std::endl; }
		if (VBO != 0) { std::cerr << "VBO of buffer was not deleted!" << std::endl; }
		if (EBO != 0) { std::cerr << "EBO of buffer was not deleted!" << std::endl; }
	}

	void cleanup()
	{
		if (VAO != 0) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
		if (VBO != 0) { glDeleteBuffers(1, &VBO); VBO = 0; }
		if (EBO != 0) { glDeleteBuffers(1, &EBO); EBO = 0; }
	}

	// updateBuffer --------------------------------------------------------------------------------------------------------

	template <typename T>
	void updateBuffer(const std::vector<T>& toBuffer, DataType dataType)
	{
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		std::size_t dataSize = toBuffer.size() * sizeof(T);

		if (elementAmount != toBuffer.size())
		{
			//std::cout << "tried to update a buffer with a different size of data" << std::endl;
			glBufferData(GL_ARRAY_BUFFER, dataSize, toBuffer.data(), GL_DYNAMIC_DRAW); // change this so gl dynamic draw is a parameter!!!!!!!!!!!!!!!!!!!!!!!!!
			elementAmount = toBuffer.size();
		}
		else
		{
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());
		}



		/*
		switch (dataType)
		{
		case pos2DlabelInt:
			if (elementAmount != toBuffer.size())
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());

			break;
		case pos2Dvel2Dcol3Dmass:
			if (elementAmount != toBuffer.size())
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());

			break;
		case pos3DCol3D:
			if (elementAmount != toBuffer.size())
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());

			break;
		case pos2DCol3D:
			if (elementAmount != toBuffer.size())
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());

			break;
		case Pos2floatAcc2floatLab1Int:
			if (elementAmount != toBuffer.size())
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, toBuffer.data());

			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
		*/
	}
	
	// buffer creation --------------------------------------------------------------------------------------------------------

	template <typename T>
	void createElementBuffer(const std::vector<T>& toBuffer, std::vector<unsigned int>& indices, DataType dataType, GLenum bufferType)
	{
		elementAmount = indices.size();

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, toBuffer.size(), toBuffer.data(), bufferType);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size(), indices.data(), bufferType);

		switch (dataType)
		{
		case pos3DNorm3DCol3DUV2D:
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(6 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
			glEnableVertexAttribArray(3);
			//glVertexAttribPointer(index, number of ->, type, GL_FALSE, size until next attribute, offsett from zero);

			//glBindBuffer(GL_ARRAY_BUFFER, 0);
			//glBindVertexArray(0);
			//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			break;
		case pos3DUV2D:
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

	template <typename T>
	void createVertexBuffer(const std::vector<T>& toBuffer, DataType dataType, GLenum bufferType)
	{
		elementAmount = toBuffer.size();

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		std::size_t dataSize = toBuffer.size() * sizeof(T);
		glBufferData(GL_ARRAY_BUFFER, dataSize, toBuffer.data(), bufferType);

		switch (dataType)
		{
		case pos2DlabelInt:
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribIPointer(1, 1, GL_INT, 3 * sizeof(float), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		case pos2Dvel2Dcol3Dmass:
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribIPointer(3, 1, GL_INT, 8 * sizeof(float), (void*)(7 * sizeof(float)));
			glEnableVertexAttribArray(3);
			break;
		case pos3D:
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			break;
		case pos3DCol3D:
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		case pos3DNOTvel3DCol3DNOTmass1D:
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)(6 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		case pos2DCol3D:
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		case pos2Dcol3Dpos2Dcol3DNOTdepth1D:
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(5 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(7 * sizeof(float)));
			glEnableVertexAttribArray(3);
			break;
		case Float2Float2Int1Int1:
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float) + 2 * sizeof(int), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float) + 2 * sizeof(int), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribIPointer(2, 1, GL_INT, 4 * sizeof(float) + 2 * sizeof(int), (void*)(4 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribIPointer(3, 1, GL_INT, 4 * sizeof(float) + 2 * sizeof(int), (void*)(4 * sizeof(float) + 1 * sizeof(int)));
			glEnableVertexAttribArray(3);
			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

	// bindVAO --------------------------------------------------------------------------------------------------------

	void BindVAO()
	{
		glBindVertexArray(VAO);
	}

private:
};