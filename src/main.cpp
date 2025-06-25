#define GLM_ENABLE_EXPERIMENTAL
#define E 2.71828182845904523536 // std::numbers::pi_v<double>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include <iostream>
#include <string>
#include "stb_image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/component_wise.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <vector>
#include <fstream>
#include <filesystem>
#include <map>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Fastor/Fastor.h>
#include <algorithm>
#include <random>
#include <format>
#include <math.h>
#include <numbers>
#include <queue>
#include <stack>


#define _CRTDBG_MAP_ALLOC
#include<iostream>
#include <crtdbg.h>

#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif


#include "openGLhelper/shader.h"
#include "cameras/camera.h"
#include "cameras/normalCamera.h"
#include "cameras/tsneCamera.h"
#include "codeData/data.h"
#include "openGLhelper/buffer.h"
#include "openGLhelper/texture.h"
#include "particles/particle3D.h"
#include "particles/particle2D.h"
#include "particles/embeddedPoint.h"
#include "particles/tsneParticle2D.h"
#include "common.h"
#include "ffthelper.h"
#include "visquad.h"
#include "openGLhelper/scene.h"
#include "nBodyInstances/tsne.h"
#include "nBodyInstances/tsneGpu.h"
#include "nBodyInstances/gravitysim.h"
#include "nBodyInstances/nBodyScenarios.h"
#include "nbodysolvers/cpu/nBodySolver.h"
#include "nbodysolvers/cpu/nBodySolverNaive.h"
#include "nbodysolvers/cpu/nBodySolverFMM.h"
#include "trees/cpu/quadtreeFMM.h"
#include "nbodysolvers/cpu/nBodySolver.h"
#include "nbodysolvers/gpu/nBodySolverGpu.h"



void framebuffer_size_callback(GLFWwindow* window, int width, int height);

unsigned int screenWidth = 1920;
unsigned int screenHeight = 1080;

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

int frameCounter = 0;
int frameCounted = 0;

int per = 0;


int main(void)
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

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
    io.IniFilename = nullptr;
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");




    {
        // global stuff
        // ---------------------------------------
        std::map<std::string, Scene*> scenes;
        std::string currentSceneName = "tsneGpu";
        //std::string currentSceneName = "tsne";




        #ifdef _WIN32
        Shader shaderLine2D("shaders/shaderLine2D.vs", "shaders/shaderLine2D.fs");
        #endif
        #ifdef linux
        Shader shaderLine2D((std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.fs").c_str());
        #endif


        // t-SNE --------------------------------------------------------------------------------------------------------------------------
        

        TSNE tsne;
        
        glm::mat4 tsneModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));
        
        #ifdef _WIN32
        Shader shaderTsne("shaders/shaderTsne.vs", "shaders/shaderTsne.fs");
        #endif
        #ifdef linux
        Shader shaderTsne((std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.fs").c_str());
        #endif
        
        tsne.nBodySelect = "FMM";
        Renderable tsneRenderablePoints(GL_POINTS, tsneModel, tsne.embeddedBuffer, &shaderTsne, nullptr);
        Renderable tsneRenderableLines(GL_LINES, tsneModel, tsne.nBodySolvers[tsne.nBodySelect]->boxBuffer, &shaderLine2D, nullptr);
        Renderable tsneRenderableForces(GL_LINES, tsneModel, tsne.forceBuffer, &shaderLine2D, nullptr);
        std::vector<Renderable> tsneRenderables{ tsneRenderablePoints, tsneRenderableLines, tsneRenderableForces };

        TsneCamera cameraTsne(glm::vec3(0.0f, 0.0f, -800.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 2.0f, 0.1f, 200.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

        Scene tsneScene("tsne", &cameraTsne, tsneRenderables);
       
        scenes[tsneScene.sceneName] = &tsneScene;
        //scenes.push_back(&tsneScene);

        
        // gravity --------------------------------------------------------------------------------------------------------------------------

        GravitySim gravitySim(10000); 
        
        glm::mat4 gravityModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

        #ifdef _WIN32
        Shader shaderGravity("shaders/shaderPos2Dvel2Dcol3D.vs", "shaders/shaderPos2Dvel2Dcol3D.fs");
        #endif
        #ifdef linux
        Shader shaderGravity((std::filesystem::current_path().parent_path().string() + "/shaders/shaderPos2Dvel2Dcol3D.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderPos2Dvel2Dcol3D.fs").c_str());
        #endif

        gravitySim.nBodySelect = "FMM";
        Renderable gravityRenderablePoints(GL_POINTS, gravityModel, gravitySim.particlesBuffer, &shaderGravity, nullptr);
        Renderable gravityRenderableLines(GL_LINES, gravityModel, gravitySim.nBodySolvers[gravitySim.nBodySelect]->boxBuffer, &shaderLine2D, nullptr);
        Renderable gravityRenderableForces(GL_LINES, gravityModel, gravitySim.forceBuffer, &shaderLine2D, nullptr);
        std::vector<Renderable> gravityRenderables{ gravityRenderablePoints, gravityRenderableLines, gravityRenderableForces };

        TsneCamera cameraGravity(glm::vec3(0.0f, 0.0f, -200.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 2.0f, 0.1f, 1000.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

        Scene gravityScene("gravity", &cameraGravity, gravityRenderables);

        scenes[gravityScene.sceneName] = &gravityScene;
        //scenes.push_back(&gravityScene);


        // gpu solver tests ------------------------------------------------------------------------------------------------------------

        TsneGpu tsneGpu;

        //tsneGpu.tests();
        
        glm::mat4 tsneGpuModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));
        
        #ifdef _WIN32
        Shader shaderTsneGpu("shaders/shaderTsne.vs", "shaders/shaderTsne.fs");
        #endif
        #ifdef linux
        Shader shaderTsneGpu((std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.vs").c_str(), (std::filesystem::current_path().parent_path().string() + "/shaders/shaderTsne.fs").c_str());
        #endif
        
        tsneGpu.nBodySelect = "naive";
        Renderable tsneGpuRenderablePoints(GL_POINTS, tsneGpuModel, tsneGpu.TsneParticlesBuffer, &shaderTsneGpu, nullptr);
        //Renderable tsneGpuRenderableLines(GL_LINES, tsneGpuModel, tsneGpu.nBodySolvers[tsneGpu.nBodySelect]->boxBuffer, &shaderLine2D, nullptr);
        Renderable tsneGpuRenderableForces(GL_LINES, tsneGpuModel, tsneGpu.forceBuffer, &shaderLine2D, nullptr);
        std::vector<Renderable> tsneGpuRenderables{ tsneGpuRenderablePoints, tsneGpuRenderableForces };

        TsneCamera cameraTsneGpu(glm::vec3(0.0f, 0.0f, -800.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 2.0f, 0.1f, 200.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

        Scene tsneGpuScene("tsneGpu", &cameraTsneGpu, tsneGpuRenderables);
       
        scenes[tsneGpuScene.sceneName] = &tsneGpuScene;
        
        // sceneNames --------------------------------------------------------------------------------------------------------------------------

        int currentSceneNameIndex = -1;
        std::vector<std::string> sceneNames;
        for (std::pair<const std::string, Scene*> scene : scenes)
        {
            sceneNames.push_back(scene.second->sceneName);
        }
        for (int i = 0; i < sceneNames.size(); i++) 
        {
            if (sceneNames[i] == currentSceneName)
            {
                currentSceneNameIndex = i;
            }
        }
        


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
            scenes[currentSceneName]->camera->processInput(window, deltaTime);
            glfwSetWindowUserPointer(window, scenes[currentSceneName]->camera);
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

            ImGui::SetNextWindowSize(ImVec2(500, 250), ImGuiCond_FirstUseEver);
            ImGui::Begin("options");

            std::string frameOutput = "frames: " + std::to_string(frameCounted);
            ImGui::Text(frameOutput.c_str());

            currentSceneName = sceneNames[currentSceneNameIndex];
            ImGui::Combo(
                "Select scene",
                &currentSceneNameIndex,
                [](void* data, int idx, const char** out_text)
                {
                    const std::vector<std::string>& vec = *static_cast<std::vector<std::string>*>(data);
                    if (idx < 0 || idx >= vec.size()) { return false; }
                    *out_text = vec[idx].c_str();
                    return true;
                },
                static_cast<void*>(&sceneNames),
                static_cast<int>(sceneNames.size())
            );
            //ImGui::Combo("Select scene", &sceneSelect, sceneNames, IM_ARRAYSIZE(sceneNames.data()));
            // this is kinda cursed, fix later!!!!!!!!!

            
            if (currentSceneName == "tsne")
            {
                if (per == 1)
                    scenes[currentSceneName]->camera->perspective = true;
                else
                    scenes[currentSceneName]->camera->perspective = false;

                //std::vector<std::string> solvers = 
                //ImGui::Combo(
                //    "Select solver",
                //    &sceneSelect,
                //    [](void* data, int idx, const char** out_text)
                //    {
                //        const std::vector<std::string>& vec = *static_cast<std::vector<std::string>*>(data);
                //        if (idx < 0 || idx >= vec.size()) { return false; }
                //        *out_text = vec[idx].c_str();
                //        return true;
                //    },
                //    static_cast<void*>(&sceneNames),
                //    static_cast<int>(sceneNames.size())
                //);
                ImGui::SliderFloat("sim speed", &tsne.timeStepsPerSec, 0.0f, 1000.0f);
                ImGui::SliderFloat("forceSize", &tsne.forceSize, 0.0f, 200.0f);
                ImGui::SliderInt("show tree level", &tsne.nBodySolvers[tsne.nBodySelect]->showLevel, -1, 10);
                ImGui::SliderInt("follow embedded points", &tsne.follow, 0, 1);

                tsne.timeStep();

                if (tsne.follow == 1)
                {
                    auto [left, right, down, up] = tsne.getEdges();
                    scenes[currentSceneName]->camera->Position = glm::vec3(left + (right - left) * 0.5f, down + (up - down) * 0.5f, scenes[currentSceneName]->camera->Position.z);
                    scenes[currentSceneName]->camera->Zoom = 1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);
                    //scenes[currentSceneName]->camera->Zoom = std::max(up - down, (right - left) / ((float)screenWidth / (float)screenHeight));
                }
            }
            else if (currentSceneName == "gravity")
            {
                ImGui::SliderFloat("sim speed", &gravitySim.timeStepsPerSec, 0.0f, 1000.0f);
                ImGui::SliderFloat("forceSize", &gravitySim.forceSize, 0.0f, 200.0f);
                ImGui::SliderInt("show tree level", &gravitySim.nBodySolvers[gravitySim.nBodySelect]->showLevel, -1, 10);

                gravitySim.timeStep();
            }
            else if (currentSceneName == "tsneGpu")
            {
                if (per == 1)
                    scenes[currentSceneName]->camera->perspective = true;
                else
                    scenes[currentSceneName]->camera->perspective = false;

                ImGui::SliderFloat("sim speed", &tsneGpu.timeStepsPerSec, 0.0f, 1000.0f);
                ImGui::SliderFloat("forceSize", &tsneGpu.forceSize, 0.0f, 200.0f);
                //ImGui::SliderInt("show tree level", &tsneGpu.nBodySolvers[tsneGpu.nBodySelect]->showLevel, -1, 10);
                ImGui::SliderInt("follow embedded points", &tsneGpu.follow, 0, 1);

                tsneGpu.timeStep();

                if (tsneGpu.follow == 1)
                {
                    auto [left, right, down, up] = tsneGpu.getEdges();
                    scenes[currentSceneName]->camera->Position = glm::vec3(left + (right - left) * 0.5f, down + (up - down) * 0.5f, scenes[currentSceneName]->camera->Position.z);
                    scenes[currentSceneName]->camera->Zoom = 1.2f * std::max((up - down) * 0.5f, (right - left) * 0.5f);
                    //scenes[currentSceneName]->camera->Zoom = std::max(up - down, (right - left) / ((float)screenWidth / (float)screenHeight));
                }
            }
            else
            {
                std::cout << "no scene selected" << std::endl;
            }









            scenes[currentSceneName]->Render();
            






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



        gravitySim.cleanup();
        tsne.cleanup();
        tsneGpu.cleanup();
        nBodyScenarios.cleanup();

        shaderTsne.cleanup();
        shaderGravity.cleanup();
        shaderTsneGpu.cleanup();
        shaderLine2D.cleanup();
    }
    
    glfwTerminate();
    _CrtDumpMemoryLeaks();
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}