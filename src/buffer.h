#ifndef BUFFER_H
#define BUFFER_H

enum BufferType {
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
	}

	Buffer(float vertices[], std::size_t verticesSize, unsigned int indices[], std::size_t indicesSize, BufferType type)
	{
		elementAmount = indicesSize / sizeof(indices[0]);

		glGenBuffers(1, &VBO);
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize, indices, GL_STATIC_DRAW);

		switch (type)
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

	Buffer(float vertices[], std::size_t verticesSize, BufferType type)
	{
		switch (type)
		{
			case pos:
				elementAmount = verticesSize / (3 * sizeof(vertices[0]));

				glGenBuffers(1, &VBO);
				glGenVertexArrays(1, &VAO);

				glBindVertexArray(VAO);

				glBindBuffer(GL_ARRAY_BUFFER, VBO);
				glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_DYNAMIC_DRAW);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
				glEnableVertexAttribArray(0);
				break;
			case posCol:
				elementAmount = verticesSize / (6 * sizeof(vertices[0]));

				glGenBuffers(1, &VBO);
				glGenVertexArrays(1, &VAO);

				glBindVertexArray(VAO);

				glBindBuffer(GL_ARRAY_BUFFER, VBO);
				glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_DYNAMIC_DRAW);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
				glEnableVertexAttribArray(1);
				break;
			default:
				std::cout << "invalid BufferType given" << std::endl;
		}
	}

	void updateBuffer(float* vertices, std::size_t verticesSize, BufferType type)
	{
		switch (type)
		{
		//case pos:
		case posCol:
			//elementAmount = verticesSize / (3 * sizeof(vertices[0]));
			
			for (int i = 0; i < verticesSize / sizeof(float); i++)
			{
				if (vertices[i] != vertices[i])
				{
					std::cout << "encountered nan" << std::endl;
					vertices[i] = 0.0f;
				}
			}

			//std::cout << glGetError() << std::endl;
			//std::cout << VAO << std::endl;
			
			glBindVertexArray(VAO);

			//std::cout << glGetError() << std::endl;

			glBindBuffer(GL_ARRAY_BUFFER, VBO);

			//std::cout << glGetError() << std::endl;

			//glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_DYNAMIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, verticesSize, vertices);

			//std::cout << glGetError() << std::endl;
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
private:

};

#endif