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
#include "nBodyScenarios.h"
#include "common.h"
#include <fstream>
#include <filesystem>
#include <map>

//std::cout << std::format("{}", std::numbers::pi_v<double>);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

unsigned int screenWidth = 1920;
unsigned int screenHeight = 1080;

std::vector<Scene*> scenes;

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

int sceneSelect = 0;
//std::string gravType = "barnesHut";
//int gravType = 0;
//int visSelect = 0;
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

    tsne.nBodySelect = "FMM";
    Renderable tsneRenderablePoints(GL_POINTS, tsneModel, tsne.embeddedBuffer, &shaderTsne, nullptr);
    Renderable tsneRenderableLines(GL_LINES, tsneModel, tsne.nBodySolvers[tsne.nBodySelect]->boxBuffer, &shaderLine2D, nullptr);
    Renderable tsneRenderableForces(GL_LINES, tsneModel, tsne.forceBuffer, &shaderLine2D, nullptr);
    std::vector<Renderable> tsneRenderables{ tsneRenderablePoints, tsneRenderableLines, tsneRenderableForces };

    Camera cameraTsne(glm::vec3(0.0f, 0.0f, -800.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 12.5, 0.1f, 200.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

    Scene tsneScene(&cameraTsne, tsneRenderables);

    scenes.push_back(&tsneScene);

    // gravity --------------------------------------------------------------------------------------------------------------------------
    GravitySim gravitySim(1000);

    glm::mat4 gravityModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

    #ifdef _WIN32
    Shader shaderGravity("shaders/shaderPos2Dvel2Dcol3D.vs", "shaders/shaderPos2Dvel2Dcol3D.fs");
    //Shader shaderLine2D("shaders/shaderLine2D.vs", "shaders/shaderLine2D.fs");
    #endif
    #ifdef linux
    Shader shaderGravity((std::filesystem::current_path().parent_path().string() + "/shaders/shaderPos2Dvel2Dcol3D.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderPos2Dvel2Dcol3D.fs").c_str());
    //Shader shaderLine2D((std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.fs").c_str());
    #endif

    gravitySim.nBodySelect = "FMM";
    Renderable gravityRenderablePoints(GL_POINTS, gravityModel, gravitySim.particlesBuffer, &shaderGravity, nullptr);
    Renderable gravityRenderableLines(GL_LINES, gravityModel, gravitySim.nBodySolvers[gravitySim.nBodySelect]->boxBuffer, &shaderLine2D, nullptr);
    Renderable gravityRenderableForces(GL_LINES, gravityModel, gravitySim.forceBuffer, &shaderLine2D, nullptr);
    std::vector<Renderable> gravityRenderables{ gravityRenderablePoints, gravityRenderableLines, gravityRenderableForces };

    Camera cameraGravity(glm::vec3(0.0f, 0.0f, -200.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 12.5, 0.1f, 1000.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

    Scene gravityScene(&cameraGravity, gravityRenderables);

    scenes.push_back(&gravityScene);

    // one time graph creation -----------------------------------------------------------------------------------------------------------

    NBodyScenarios nBodyScenarios;
    //nBodyScenarios.errorTimestepGRAVITY();
    //nBodyScenarios.errorTimestepTSNE();
     
    //nBodyScenarios.errorTimestepGRAVITYFMMtest();
     
    //nBodyScenarios.calculationtimeThetaGRAVITY();
    //nBodyScenarios.calculationtimeThetaTSNE();

    //nBodyScenarios.errorThetaGRAVITY();
    //nBodyScenarios.errorThetaTSNE();

    //nBodyScenarios.calculationtimeErrorGRAVITY();
    //nBodyScenarios.calculationtimeErrorTSNE();


    //nBodyScenarios.testNodeNode();

    std::cout << "im done with nBodyScenarios!" << std::endl;

    //scene creation done ----------------------------------------------------------------------------------------------------------------


    float lastTimePressed = 0.0f;
    float lastFrameUpdate = 0.0f;

    // render loop
    // -----------
    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0f);
    while (!glfwWindowShouldClose(window))
    {
        // initial
        scenes[sceneSelect]->camera->processInput(window, deltaTime);
        glfwSetWindowUserPointer(window, scenes[sceneSelect]->camera);
        glfwSetCursorPosCallback(window, Camera::mouse_callback);
        glfwSetScrollCallback(window, Camera::scroll_callback);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float timeBeginFrame = glfwGetTime();
        // initial


        frameCounter++;
        if (glfwGetTime() - lastFrameUpdate > 1.0f)
        {
            lastFrameUpdate = glfwGetTime();
            frameCounted = frameCounter;
            frameCounter = 0;
        }

        ImGui::Begin("options");

        std::string frameOutput = "frames: " + std::to_string(frameCounted);
        ImGui::Text(frameOutput.c_str());

        
        if (sceneSelect == 0)
        {
            ImGui::SliderInt("show tree level", &tsne.nBodySolvers[tsne.nBodySelect]->showLevel, -1, 10);
            ImGui::SliderInt("follow embedded points", &tsne.follow, 0, 1);

            tsne.timeStep();

            if (tsne.follow == 1)
            {
                auto [left, right, down, up] = tsne.getEdges();
                scenes[0]->camera->Position = glm::vec3(left + (right - left) * 0.5f, down + (up - down) * 0.5f, scenes[0]->camera->Position.z);
                scenes[0]->camera->Zoom = 1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);
                //std::cout << "horizontal size: " << right - left << std::endl;
                //std::cout << "vertical size:   " << up - down << std::endl;

                //scenes[0]->camera->Zoom = std::max(up - down, (right - left) / ((float)screenWidth / (float)screenHeight));
            }
        }
        else
        {
            ImGui::SliderInt("show tree level", &gravitySim.nBodySolvers[gravitySim.nBodySelect]->showLevel, -1, 10);

            gravitySim.timeStep();
        }





        if (per == 1)
            scenes[0]->camera->perspective = true;
        else
            scenes[0]->camera->perspective = false;
        


        scenes[sceneSelect]->Render();

        

        



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
    gravitySim.cleanup();
    shaderGravity.cleanup();
    shaderTsne.cleanup();
    shaderLine2D.cleanup();
    nBodyScenarios.cleanup();

    glfwTerminate();
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}