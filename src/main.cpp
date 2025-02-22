#define GLM_ENABLE_EXPERIMENTAL
#define E 2.71828182845904523536


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include "shader.h"
#include <string>
#include "stb_image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include "camera.h"
#include "data.h"
#include "buffer.h"
#include "texture.h"
#include "particles/particle3D.h"
#include "particles/particle2D.h"
#include "particles/embeddedPoint.h"
#include <vector>
#include "nbodySim.h"
#include "ffthelper.h"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include "visquad.h"
#include "scene.h"
#include "tsne.h"
#include "gravitysim.h"
#include "common.h"
#include <fstream>
#include <filesystem>

//std::cout << std::format("{}", std::numbers::pi_v<double>);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

unsigned int screenWidth = 1920;
unsigned int screenHeight = 1080;

std::vector<Scene*> scenes;

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

//std::string gravType = "barnesHut";
int gravType = 0;
int visSelect = 0;
int frameCounter = 0;
int frameCounted = 0;

int per = 0;


int main(void)
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "master thesis", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMaximizeWindow(window);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0);//unlimited frames!!!
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))//load glad
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");






    // global stuff
    // ---------------------------------------

    /*
    NbodySim simulation(rainbowCube, naive, 1.0f, 9999999, 250, 0.0001f, 0.001f, 1.0f, 30, 0);
    glm::mat4 particleModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));
    Shader shaderPoint("shaders/shaderPoint.vs", "shaders/shaderPoint.fs");
    Shader shaderLine("shaders/shaderLine.vs", "shaders/shaderLine.fs");

    Renderable particleRenderable(GL_POINTS, particleModel, simulation.particlesBuffer, &shaderPoint, nullptr);
    Renderable boxRenderable(GL_LINES, particleModel, simulation.boxBuffer, &shaderLine, nullptr);
    Renderable* particleBoxRenderables = new Renderable[2]{ particleRenderable, boxRenderable };
    Renderable* particleRenderables = new Renderable[1]{ particleRenderable };

    Camera cameraNbody(glm::vec3(0.0f, 0.0f, -30.0f), glm::vec3(0.0f,1.0f,0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 12.5, 0.1f, 45.0f, 0.001f, 1000.0f, true, &screenWidth, &screenHeight);

    Scene nbodyParticleBoxScene(&cameraNbody, particleBoxRenderables, 2 * sizeof(Renderable));
    Scene nbodyParticleScene(&cameraNbody, particleRenderables, 1 * sizeof(Renderable));


    VisQuad potSimulation(16 * 8, 9 * 8, 0.1f, 30, 1.0f, 1.0f);
    Buffer quad(verticesQuad, sizeof(verticesQuad), indicesQuad, sizeof(indicesQuad), pos3DUV2D, GL_STATIC_DRAW);
    Shader shaderQuad("shaders/shaderQuad.vs", "shaders/shaderQuad.fs");
    Renderable quadRenderable(GL_TRIANGLES, glm::mat4(1.0f), &quad, &shaderQuad, potSimulation.texture);
    Renderable* quadRenderables = new Renderable[1]{ quadRenderable };

    Camera cameraQuad(glm::vec3(0.0f, 0.0f, -130.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 12.5, 0.1f, 45.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

    Scene quadScene(&cameraQuad, quadRenderables, 1 * sizeof(Renderable));

    
    scenes[0] = &nbodyParticleBoxScene;
    scenes[1] = &nbodyParticleScene;
    scenes[2] = &quadScene;
    */
    //NbodySim simulation(rainbowCube, naive, 1.0f, 9999999, 250, 0.0001f, 0.001f, 1.0f, 30, 0);
    
    // t-SNE --------------------------------------------------------------------------------------------------------------------------
    TSNE tsne;
    
    glm::mat4 tsneModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

    #ifdef _WIN32
    Shader shaderTsne("shaders/shaderTsne.vs", "shaders/shaderTsne.fs");
    Shader shaderLine2D("shaders/shaderLine2D.vs", "shaders/shaderLine2D.fs");
    #endif
    #ifdef linux
    Shader shaderTsne((std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.fs").c_str());
    Shader shaderLine2D((std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.fs").c_str());
    #endif

    Renderable tsneRenderablePoints(GL_POINTS, tsneModel, tsne.embeddedBuffer, &shaderTsne, nullptr);
    Renderable tsneRenderableLines(GL_LINES, tsneModel, tsne.nBodySolverBarnesHut.boxBuffer, &shaderLine2D, nullptr);
    Renderable* tsneRenderables = new Renderable[2]{ tsneRenderablePoints, tsneRenderableLines };

    Camera cameraTsne(glm::vec3(0.0f, 0.0f, -800.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 12.5, 0.1f, 200.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

    Scene tsneScene(&cameraTsne, tsneRenderables, 2 * sizeof(Renderable));

    scenes.push_back(&tsneScene);

    // gravity --------------------------------------------------------------------------------------------------------------------------






    float lastTimePressed = 0.0f;
    float lastFrameUpdate = 0.0f;

    // render loop
    // -----------
    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0f);
    while (!glfwWindowShouldClose(window))
    {
        // initial
        scenes[0]->camera->processInput(window, deltaTime);

        glfwSetWindowUserPointer(window, scenes[0]->camera);
        glfwSetCursorPosCallback(window, Camera::mouse_callback);
        glfwSetScrollCallback(window, Camera::scroll_callback);

        //glfwSetScrollCallback(window, scroll_callback);
        //processInput(window);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float timeBeginFrame = glfwGetTime();

        tsne.timeStep();
        
        auto [left, right, down, up] = tsne.getEdges();
        scenes[0]->camera->Position = glm::vec3(left + (right - left) * 0.5f, down + (up - down) * 0.5f, scenes[0]->camera->Position.z);
        scenes[0]->camera->Zoom = 1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);
        //std::cout << "horizontal size: " << right - left << std::endl;
        //std::cout << "vertical size:   " << up - down << std::endl;

        //scenes[0]->camera->Zoom = std::max(up - down, (right - left) / ((float)screenWidth / (float)screenHeight));
        
        scenes[0]->Render();

        
        if (per == 1)
        {
            scenes[0]->camera->perspective = true;
        }
        else
        {
            scenes[0]->camera->perspective = false;
        }
        


        /*
        if (timeBeginFrame - lastTimePressed > 0.2f)
        {
            if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
                simulation.showLevel += 1;
            if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
                simulation.showLevel -= 1;

            lastTimePressed = timeBeginFrame;
        }

        if (visSelect == 0)
        { 
            if (gravType == 0)
            {
                simulation.setAccelerationType(barnesHut);
                simulation.simulate();
                //std::cout << simulation.lineSegments.size() * 2 << std::endl;
                scenes[0]->Render();
            }
            else
            {
                simulation.setAccelerationType(naive);
                simulation.simulate();

                scenes[1]->Render();
            }
        }
        else
        {
            potSimulation.updateVisual();
            scenes[2]->Render();
        }
        */

        


        frameCounter++;
        if (glfwGetTime() - lastFrameUpdate > 1.0f)
        {
            lastFrameUpdate = glfwGetTime();
            frameCounted = frameCounter;
            frameCounter = 0;
        }

        std::string frameString = std::to_string(frameCounted);
        std::string frame = "frames: ";
        std::string frameOutput = frame + frameString;
        
        
        ImGui::Begin("options");
        ImGui::Text(frameOutput.c_str());
        //static float myVariable = 0.0f;
        //ImGui::SliderFloat("My Variable", &myVariable, 0.0f, 1.0f);

        ImGui::SliderInt("orthographic <-> perspective", &per, 0, 1);
        
        //ImGui::SliderInt("gravitySim <-> potentialSolver", &visSelect, 0, 1);
        //if (visSelect == 0)
        //{ 
        //    ImGui::SliderInt("Barnes&Hut <-> naive: ", &gravType, 0, 1);
        //    if (gravType == 0)
        //    {
        //        ImGui::SliderInt("show tree level", &simulation.showLevel, 0, 10);
        //    }
        //}
        
        ImGui::SliderInt("show tree level", &tsne.nBodySolverBarnesHut.showLevel, 0, 10);

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        

        // final
        glfwSwapBuffers(window);
        glfwPollEvents();
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        // final
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    

    tsne.cleanup();
    shaderTsne.cleanup();
    shaderLine2D.cleanup();

    glfwTerminate();
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}

/*
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    scenes[std::max(2 * visSelect, gravType)]->camera->processMouseScroll(static_cast<float>(yoffset));
}
*/