#ifndef SCENE_H
#define SCENE_H

#include "camera.h"
#include "shader.h"
#include "buffer.h"
#include "texture.h"

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

class Scene
{
public:
	Camera* camera;
	glm::mat4 view;
	glm::mat4 projection;
	Renderable* renderables;

	Scene()
	{
	}

	Scene(Camera* initCamera, Shader* initShader, glm::mat4 initModel, Renderable* initRenderables)
	{
		camera = initCamera;
		view = glm::mat4(0.0f);
		projection = glm::mat4(0.0f);
		renderables = initRenderables;
	}

	~Scene()
	{
		delete[] renderables;
	}

private:

};
#endif