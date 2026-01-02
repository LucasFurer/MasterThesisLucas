#define GLM_ENABLE_EXPERIMENTAL

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <glm/glm.hpp>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <vector>
#include <iostream>
#include <map>
#include <filesystem>
#include "stb_image.h" // hmmm change this
#include <chrono>

#include "openGLhelper/shader.h"
#include "openGLhelper/scene.h"
#include "openGLhelper/buffer.h"
#include "openGLhelper/texture.h"
#include "cameras/camera.h"
#include "cameras/normalCamera.h"
#include "cameras/tsneCamera.h"
#include "nBodyInstances/tsne.h"
#include "nBodyInstances/tsneGpu.h"
#include "nBodyInstances/gravitysim.h"
#include "nBodyInstances/nBodyScenarios.h"
#include "nBodyInstances/tsneTests.h"
#include "common.h"
#include "codeData/data.h"
#include "visualization/multipoleVis.h"
#include "Timer.h"

//#define _CRTDBG_MAP_ALLOC
//#include <iostream>
//#include <crtdbg.h>
//
//#ifdef _DEBUG
//#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
//#define new DEBUG_NEW
//#endif


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
    //_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

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
        //std::string currentSceneName = "tsneGpu";
        std::string currentSceneName = "tsne";


        

        #ifdef _WIN32
        Shader shaderLine2D
        (
            ((std::filesystem::current_path() / "shaders" / "shaderLine2D.vs").string()).c_str(),
            ((std::filesystem::current_path() / "shaders" / "shaderLine2D.fs").string()).c_str()
        );
        #endif
        #ifdef linux
        Shader shaderLine2D
        (
            (std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.vs").c_str(), 
            (std::filesystem::current_path().parent_path().string() + "/shaders/shaderLine2D.fs").c_str()
        );
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
        
        tsne.nBodySelect = "FMM_SYM_MORTON";
        //tsne.nBodySelect = "FMM_MORTON";
        //tsne.nBodySelect = "FMM";
        //tsne.nBodySelect = "PM";
        //tsne.nBodySelect = "BH";
        //tsne.nBodySelect = "naive";
        Renderable tsneRenderablePoints(GL_POINTS, tsneModel, tsne.embeddedBuffer, &shaderTsne, nullptr);
        Renderable tsneRenderableLines(GL_LINES, tsneModel, tsne.nodeBuffer, &shaderLine2D, nullptr);
        Renderable tsneRenderableForces(GL_LINES, tsneModel, tsne.forceBuffer, &shaderLine2D, nullptr);
        std::vector<Renderable> tsneRenderables{ tsneRenderablePoints, tsneRenderableLines, tsneRenderableForces };

        TsneCamera cameraTsne(glm::vec3(0.0f, 0.0f, -800.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f, glm::vec3(0.0f, 0.0f, -1.0f), 2.0f, 0.1f, 200.0f, 0.001f, 1000.0f, false, &screenWidth, &screenHeight);

        Scene tsneScene("tsne", &cameraTsne, tsneRenderables);
       
        scenes[tsneScene.sceneName] = &tsneScene;

        
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

        TsneTest tsne_test;
        tsne_test.errorTimestepTSNE("MNIST_digits", 10000, 30.0f, -1.0f, 1000, 0.5f, -1, 1.0f, 296343u);
        //tsne_test.calculationtimeThetaTSNE("MNIST_digits", 10000, 30.0f, -1.0f, 10, 0.3f, 6, 1.0f, 5, 1.0, 296343u);
        //tsne_test.errorThetaTSNE("MNIST_digits", 70000, 30.0f, -1.0f, 2, 0.1f, 20, 2.0f, 0.1f, 2.0, 296343u);
        //tsne_test.calculationtimeErrorTSNE("MNIST_digits", 70000, 30.0f, -1.0f, 1, 0.0f, 20, 2.0f, 0.1f, 2.0f, 296343u);

        //NBodyScenarios nBodyScenarios;
        //nBodyScenarios.errorTimestepTSNE("MNIST_digits", 1000, 500, 1.0f, 30.0f);
        //std::cout << "starting tests--------------------------" << std::endl;

        //std::vector<float> perpValues{};
        //for (float val : perpValues)
        //{
        //    float perp = val;
        //    std::string dataSet = "MNIST_digits";
        //    int dataSize = 1000;

        //    std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
        //    nBodyScenarios.errorTimestepTSNE(dataSet, dataSize, 1000, 1.0f, perp); // "MNIST_digits" 10000 1000 1.0f 5.0f => 523 sec
        //    std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();
        //    std::chrono::duration<double> elapsed = end - start;
        //    std::cout << "errorTimestepTSNE test done in: " << elapsed.count() << std::endl;
        //    /*
        //    start = std::chrono::high_resolution_clock::now();
        //    nBodyScenarios.calculationtimeThetaTSNE(dataSet, dataSize, 100, perp); // "MNIST_digits" 10000 100 5.0f => 882 sec
        //    end = std::chrono::high_resolution_clock::now();
        //    elapsed = end - start;
        //    std::cout << "calculationtimeThetaTSNE test done in: " << elapsed.count() << std::endl;

        //    start = std::chrono::high_resolution_clock::now();
        //    nBodyScenarios.errorThetaTSNE(dataSet, dataSize, 100, perp); // "MNIST_digits" 10000 100 5.0f => 825 sec
        //    end = std::chrono::high_resolution_clock::now();
        //    elapsed = end - start;
        //    std::cout << "errorThetaTSNE test done in: " << elapsed.count() << std::endl;

        //    start = std::chrono::high_resolution_clock::now();
        //    nBodyScenarios.calculationtimeErrorTSNE(dataSet, dataSize, 100, perp); // "MNIST_digits" 10000 100 5.0f => 885 sec
        //    end = std::chrono::high_resolution_clock::now();
        //    elapsed = end - start;
        //    std::cout << "calculationtimeErrorTSNE test done in: " << elapsed.count() << std::endl;
        //    */
        //}
        //std::cout << "all test done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!--------------------------" << std::endl;


        //nBodyScenarios.errorTimestepGRAVITY();
        //nBodyScenarios.errorTimestepGRAVITYFMMtest();
        //nBodyScenarios.calculationtimeThetaGRAVITY();
        //nBodyScenarios.errorThetaGRAVITY();
        //nBodyScenarios.calculationtimeErrorGRAVITY();

        //nBodyScenarios.testNodeNode();

        //std::cout << "im done with nBodyScenarios!" << std::endl;
        //MultipoleVis::initMultipoleVisData();
        //MultipoleVis::testFMMtoBH();



        

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

                std::string frameOutput = "iteration: " + std::to_string(tsne.iteration_counter);
                ImGui::Text(frameOutput.c_str());

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
                ImGui::SliderFloat("sim speed", &tsne.desired_iteration_per_second, 0.0f, 1000.0f);
                ImGui::SliderFloat("forceSize", &tsne.forceSize, 0.0f, 200.0f);
                ImGui::SliderInt("show tree level", &tsne.nodeLevelToShow, -1, 10);
                ImGui::SliderInt("follow embedded points", &tsne.follow, 0, 1);


                tsne.timeStep();


                if (tsne.follow == 1)
                {
                    //auto [left, right, down, up] = tsne.getEdges();
                    float left = tsne.minPos.x;
                    float down = tsne.minPos.y;
                    float right = tsne.maxPos.x;
                    float up = tsne.maxPos.y;
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



        //gravitySim.cleanup();
        tsne.cleanup();
        //tsneGpu.cleanup();
        //nBodyScenarios.cleanup();

        shaderTsne.cleanup();
        //shaderGravity.cleanup();
        //shaderTsneGpu.cleanup();
        shaderLine2D.cleanup();
    }
    
    glfwTerminate();
    //_CrtDumpMemoryLeaks();
    return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}