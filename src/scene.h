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
	std::vector<Renderable> renderables;

	Scene()
	{
	}

	Scene(Camera* initCamera, std::vector<Renderable> initRenderables)
	{
		camera = initCamera;
		renderables = initRenderables;
	}

	~Scene()
	{

	}

	void Render()
	{
		for (int i = 0; i < renderables.size(); i++)
		{
			renderables[i].shader->use();

			switch (renderables[i].renderType)
			{
			case GL_POINTS:
				renderables[i].shader->setMat4("view", camera->getViewMatrix());
				renderables[i].shader->setMat4("projection", camera->getProjectionMatrix());
				renderables[i].shader->setMat4("model", renderables[i].model);

				renderables[i].buffer->BindVAO();
				
				glDrawArrays(GL_POINTS, 0, renderables[i].buffer->elementAmount);
				break;
			case GL_LINES:
				renderables[i].shader->setMat4("view", camera->getViewMatrix());
				renderables[i].shader->setMat4("projection", camera->getProjectionMatrix());
				renderables[i].shader->setMat4("model", renderables[i].model);

				renderables[i].buffer->BindVAO();

				glDrawArrays(GL_LINES, 0, renderables[i].buffer->elementAmount);
				break;
			case GL_TRIANGLES:
				renderables[i].shader->setInt("planeTexture", 0);
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, renderables[i].texture->TEX);

				//renderables[i].shader->use();

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