#ifndef BUFFER_H
#define BUFFER_H

//#include "common.h"
#include "particles/embeddedPoint.h"

enum DataType
{
	pos3DNorm3DCol3DUV2D,
	pos3D,
	pos3DCol3D,
	pos3DUV2D,
	pos2DCol3D,
	pos2DlabelInt,
	pos2Dvel2Dcol3Dmass
};

class Buffer
{
public:
	unsigned int VBO;
	unsigned int VAO;
	unsigned int EBO;
	int elementAmount;

	Buffer()
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);
		elementAmount = 0;
	}

	Buffer(float vertices[], std::size_t verticesSize, unsigned int indices[], std::size_t indicesSize, DataType dataType, GLenum bufferType) // deprecated
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);

		createElementBuffer(vertices, verticesSize, indices, indicesSize, dataType, bufferType);
	}

	Buffer(float vertices[], std::size_t verticesSize, DataType dataType, GLenum bufferType) // deprecated
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);

		createVertexBuffer(vertices, verticesSize, dataType, bufferType);
	}

	template <typename T>
	Buffer(T* data, int dataAmount, DataType dataType, GLenum bufferType)
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);

		createVertexBufferNew(data, dataAmount, dataType, bufferType);
	}

	void updateBuffer(float* vertices, std::size_t verticesSize, DataType dataType)
	{
		switch (dataType)
		{
		//case pos:
		case pos3DCol3D:
			if (elementAmount != verticesSize / (6 * sizeof(float)))
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			for (int i = 0; i < verticesSize / sizeof(float); i++)
			{
				if (vertices[i] != vertices[i])
				{
					std::cout << "encountered nan" << std::endl;
					vertices[i] = 0.0f;
				}
			}
			
			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, verticesSize, vertices);

			break;
		case pos2DCol3D:
			if (elementAmount != verticesSize / (5 * sizeof(float)))
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			for (int i = 0; i < verticesSize / sizeof(float); i++)
			{
				if (vertices[i] != vertices[i])
				{
					std::cout << "encountered nan" << std::endl;
					vertices[i] = 0.0f;
				}
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, verticesSize, vertices);

			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

	template <typename T>
	void updateBufferNew(T* data, int dataAmount, DataType dataType)
	{
		std::size_t dataSize = dataAmount * sizeof(T);

		switch (dataType)
		{
		case pos2DlabelInt:
		
			if (elementAmount != dataAmount)
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);

			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, data);

			break;

		case pos2Dvel2Dcol3Dmass:

			if (elementAmount != dataAmount)
			{
				std::cout << "tried to update a buffer with a different size of data" << std::endl;
			}

			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);

			glBufferSubData(GL_ARRAY_BUFFER, 0, dataSize, data);

			break;
		
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

	void BindVAO()
	{
		glBindVertexArray(VAO);
	}

	~Buffer()
	{
		//if (VAO != 0) { glDeleteVertexArrays(1, &VAO); }
		//if (VBO != 0) { glDeleteBuffers(1, &VBO); }
		//if (EBO != 0) { glDeleteBuffers(1, &EBO); }
	}

	void cleanup()
	{
		if (VAO != 0) { glDeleteVertexArrays(1, &VAO); }
		if (VBO != 0) { glDeleteBuffers(1, &VBO); }
		if (EBO != 0) { glDeleteBuffers(1, &EBO); }
	}


	void createElementBuffer(float vertices[], std::size_t verticesSize, unsigned int indices[], std::size_t indicesSize, DataType dataType, GLenum bufferType)
	{
		elementAmount = indicesSize / sizeof(indices[0]);

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, bufferType);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize, indices, bufferType);

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

	void createVertexBuffer(float vertices[], std::size_t verticesSize, DataType dataType, GLenum bufferType)
	{
		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, bufferType);

		switch (dataType)
		{
		case pos3D:
			elementAmount = verticesSize / (3 * sizeof(vertices[0]));

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			break;
		case pos3DCol3D:
			elementAmount = verticesSize / (6 * sizeof(vertices[0]));

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		case pos2DCol3D:
			elementAmount = verticesSize / (5 * sizeof(vertices[0]));

			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

	template <typename T>
	void createVertexBufferNew(T* data, int dataAmount, DataType dataType, GLenum bufferType)
	{
		elementAmount = dataAmount;

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		//std::size_t dataSize = dataAmount * (2 * sizeof(float) + sizeof(int));
		std::size_t dataSize = dataAmount * sizeof(T);
		glBufferData(GL_ARRAY_BUFFER, dataSize, data, bufferType);

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
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

private:
};

#endif