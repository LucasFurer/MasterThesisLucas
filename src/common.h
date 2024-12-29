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

#endif
