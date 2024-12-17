#ifndef SCENE_H
#define SCENE_H

#include "camera.h"
#include "shader.h"
#include "buffer.h"
#include "texture.h"
#include "common.h"

class Scene
{
public:
	Camera* camera;
	glm::mat4 view;
	glm::mat4 projection;
	unsigned int* screenWidth;
	unsigned int* screenHeight;
	Renderable* renderables;
	std::size_t renderablesSize;

	Scene()
	{
	}

	Scene(Camera* initCamera, unsigned int* screenWidthReference, unsigned int* screenHeightReference, Renderable* initRenderables, std::size_t initRenderablesSize)
	{
		camera = initCamera;
		view = initCamera->getViewMatrix();
		screenWidth = screenWidthReference;
		screenHeight = screenHeightReference;
		projection = glm::perspective(glm::radians(initCamera->Zoom), (float)*screenWidthReference / (float)*screenHeightReference, 0.01f, 1000.0f);
		renderables = initRenderables;
		renderablesSize = initRenderablesSize;
	}

	~Scene()
	{
		delete[] renderables;
	}

	void Render()
	{
		for (int i = 0; i < renderablesSize / sizeof(Renderable); i++)
		{
			renderables[i].shader->use();

			switch (renderables[i].renderType)
			{
			case GL_POINTS:
				view = camera->getViewMatrix();
				renderables[i].shader->setMat4("view", view);

				projection = glm::perspective(glm::radians(camera->Zoom), (float)*screenWidth / (float)*screenHeight, 0.01f, 1000.0f);
				renderables[i].shader->setMat4("projection", projection);

				renderables[i].shader->setMat4("model", renderables[i].model);

				renderables[i].buffer->BindVAO();
				
				glDrawArrays(GL_POINTS, 0, renderables[i].buffer->elementAmount);
				break;
			case GL_LINES:

				break;
			case GL_TRIANGLES:
				renderables[i].shader->setInt("planeTexture", 0);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, renderables[i].texture->TEX);
				renderables[i].shader->use();

				renderables[i].buffer->BindVAO();
				glDrawElements(GL_TRIANGLES, renderables[i].buffer->elementAmount, GL_UNSIGNED_INT, 0);
				break;
			default:
				std::cout << "invalid render type" << std::endl;
			}

		}
	}

private:

};
#endif