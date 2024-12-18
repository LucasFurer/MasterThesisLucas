#ifndef BUFFER_H
#define BUFFER_H

enum DataType {
	posNormColUV,
	pos,
	posCol,
	posUV
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

	Buffer(float vertices[], std::size_t verticesSize, unsigned int indices[], std::size_t indicesSize, DataType dataType, GLenum bufferType)
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);

		createElementBuffer(vertices, verticesSize, indices, indicesSize, dataType, bufferType);
	}

	Buffer(float vertices[], std::size_t verticesSize, DataType dataType, GLenum bufferType)
	{
		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);

		createVertexBuffer(vertices, verticesSize, dataType, bufferType);
	}

	void updateBuffer(float* vertices, std::size_t verticesSize, DataType dataType)
	{
		switch (dataType)
		{
		//case pos:
		case posCol:
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
		case posNormColUV:
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
		case posUV:
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
		case pos:
			elementAmount = verticesSize / (3 * sizeof(vertices[0]));

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			break;
		case posCol:
			elementAmount = verticesSize / (6 * sizeof(vertices[0]));

			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			break;
		default:
			std::cout << "invalid BufferType given" << std::endl;
		}
	}

private:
};

#endif