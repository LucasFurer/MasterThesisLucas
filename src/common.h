#ifndef ENUMS_H
#define ENUMS_H

enum AccelerationType 
{
    naive,
    barnesHut,
    particleMesh
};

struct Renderable
{
	GLenum renderType;
	glm::mat4 model;
	Buffer* buffer;
	Shader* shader;
	Texture* texture;

	Renderable(GLenum initRenderType, glm::mat4 initModel, Buffer* initbuffer, Shader* initShader, Texture* initTexture)
	{
		renderType = initRenderType;
		model = initModel;
		buffer = initbuffer;
		shader = initShader;
		texture = initTexture;
	}
};

struct VertexPos2Col3
{
	glm::vec2 position;
	glm::vec3 color;

	VertexPos2Col3(glm::vec2 initPosition, glm::vec3 initColor)
	{
		position = initPosition;
		color = initColor;
	}

	template <typename T>
	static std::vector<VertexPos2Col3> particlesAccelerationsToVertexPos2Col3(const std::vector<T>& particles, std::vector<glm::vec2>& accelerations)
	{
		std::vector<VertexPos2Col3> result;
		for (int i = 0; i < particles.size(); i++)
		{
			glm::vec2 linePosB = particles[i].position;
			glm::vec3 lineColB = glm::vec3(1.0f, 0.0f, 0.0f);

			glm::vec2 linePosE = particles[i].position + 10.0f * accelerations[i];
			//glm::vec2 linePosE = particles[i].position + particles[i].speed;
			glm::vec3 lineColE = glm::vec3(1.0f, 0.0f, 0.0f);

			result.push_back(VertexPos2Col3(linePosB, lineColB));
			result.push_back(VertexPos2Col3(linePosE, lineColE));
		}
		return result;
	}
};

struct LineSegment2D
{
	glm::vec2 pointB;
	glm::vec3 colorB;
	glm::vec2 pointE;
	glm::vec3 colorE;
	int depth;

	LineSegment2D(glm::vec2 initPointB, glm::vec2 initPointA, glm::vec3 initColorB, glm::vec3 initColorE, int initDepth)
	{
		pointB = initPointB;
		pointE = initPointA;
		colorB = initColorB;
		colorE = initColorE;
		depth = initDepth;
	}

	LineSegment2D()
	{

	}

	static std::vector<VertexPos2Col3> LineSegmentToVertexPos2Col3(const std::vector<LineSegment2D>& lineSegments)
	{
		std::vector<VertexPos2Col3> result;
		for (LineSegment2D lineSegment : lineSegments)
		{
			result.push_back(VertexPos2Col3(lineSegment.pointB, lineSegment.colorB));
			result.push_back(VertexPos2Col3(lineSegment.pointE, lineSegment.colorE));
		}
		return result;
	}
};

#endif
