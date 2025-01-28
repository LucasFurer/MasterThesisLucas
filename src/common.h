#ifndef ENUMS_H
#define ENUMS_H

float E = 2.71828182845904523536;

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

struct LineSegment2D
{
	glm::vec2 pointB;
	glm::vec2 pointE;
	glm::vec3 colorB;
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

	static float* LineSegmentToFloat(LineSegment2D* lineSegments, std::size_t lineSegmentsSize)
	{
		int lineSegmentAmount = lineSegmentsSize / sizeof(LineSegment2D);

		float* result = new float[10 * lineSegmentAmount];

		for (int i = 0; i < lineSegmentAmount; i++)
		{
			result[10 * i + 0] = lineSegments[i].pointB.x;
			result[10 * i + 1] = lineSegments[i].pointB.y;

			result[10 * i + 2] = lineSegments[i].colorB.r;
			result[10 * i + 3] = lineSegments[i].colorB.g;
			result[10 * i + 4] = lineSegments[i].colorB.b;

			result[10 * i + 5] = lineSegments[i].pointE.x;
			result[10 * i + 6] = lineSegments[i].pointE.y;

			result[10 * i + 7] = lineSegments[i].colorE.r;
			result[10 * i + 8] = lineSegments[i].colorE.g;
			result[10 * i + 9] = lineSegments[i].colorE.b;
		}

		return result;
	}
};

#endif
