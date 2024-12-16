#define GLM_ENABLE_EXPERIMENTAL
//#define E 2.71828182845904523536;


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
#include "particle.h"
#include <vector>
#include "nbodySim.h"
#include "ffthelper.h"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include "visquad.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

unsigned int screenWidth = 1920;
unsigned int screenHeight = 1080;

Camera camera(glm::vec3(0.0f, 0.0f, -130.0f));
float lastX = 400;
float lastY = 300;
bool firstMouse = true;

glm::vec3 lightPos(-4.0f, 2.0f, -1.5f);

float deltaTime = 0.0f;	// Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame

//std::string gravType = "barnesHut";
int gravType = 0;
int frameCounter = 0;
int frameCounted = 0;
int visSelect = 0;

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
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
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
    NbodySim simulation = NbodySim(rainbowCube, naive, 1.0f, 9999999, 1500, 0.0001f, 0.001f, 1.0f, 30, 0);
    VisQuad potSimulation(16 * 8, 9 * 8, 0.1f, 30, 1.0f, 1.0f);
    //MoldPlane simulation(1920 / 4, 1080 / 4, 50000, 0.7f, 3.0f, 0.1f, 60);


    std::vector<Buffer*> objects = { simulation.particlesBuffer };
    Buffer quad(verticesQuad, sizeof(verticesQuad), indicesQuad, sizeof(indicesQuad), posUV);
    std::vector<Buffer*> potObjects = { &quad };
    
    glm::mat4 particleModel = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));
    std::vector<glm::mat4> transforms = { particleModel };

    Shader shaderQuad("shaders/shaderQuad.vs", "shaders/shaderQuad.fs");
    Shader shader("shaders/shader.vs", "shaders/shader.fs");
    shader.use();
    //Shader shaderQuad("shaders/shaderQuad.vs", "shaders/shaderQuad.fs");
    //shaderQuad.use();
   
    float prevVal = 0.0f;
    float lastTimePressed = 0.0f;

    float lastFrameUpdate = 0.0f;



    // render loop
    // -----------
    glEnable(GL_DEPTH_TEST);
    glPointSize(4.0f);
    while (!glfwWindowShouldClose(window))
    {
        if (glfwGetTime() - lastTimePressed > 0.2f)
        {
            if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
                simulation.showLevel += 1;
            if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
                simulation.showLevel -= 1;

            lastTimePressed = glfwGetTime();
        }


        // initial
        processInput(window);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        float timeValue = glfwGetTime();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // initial

        

        if (visSelect == 0)
        { 
            //std::cout << "total time for frame: " << timeValue - prevVal << std::endl;
            prevVal = timeValue;

            if (gravType == 0)
            {
                //std::cout << "im in barnes hut" << std::endl;
                simulation.setAccelerationType(barnesHut);
                simulation.simulate();
            }
            else
            {
                simulation.setAccelerationType(naive);
                simulation.simulate();
            }





            shader.use();

            glm::mat4 view = camera.GetViewMatrix();
            shader.setMat4("view", view);

            glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)screenWidth / (float)screenHeight, 0.01f, 1000.0f);
            shader.setMat4("projection", projection);

        

            for (Buffer* buff : objects)
            {
                shader.setMat4("model", transforms[0]);

                objects[0]->BindVAO();
                glDrawArrays(GL_POINTS, 0, simulation.particlesSize/sizeof(Particle));

                //buff->BindVAO();
                //glDrawElements(GL_TRIANGLES, buff->elementAmount, GL_UNSIGNED_INT, 0);
            }

            if (gravType == 0)
            {
                simulation.boxBuffer->BindVAO();
                glDrawArrays(GL_LINES, 0, simulation.lineSegments.size() * 2);
            }

        }
        else
        {
            shaderQuad.use();
            potSimulation.updateVisual();

            shaderQuad.setInt("planeTexture", 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, potSimulation.texture->TEX);
            shaderQuad.use();

            for (Buffer* buff : potObjects)
            {
                buff->BindVAO();
                glDrawElements(GL_TRIANGLES, buff->elementAmount, GL_UNSIGNED_INT, 0);
            }
        }

        




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

        ImGui::SliderInt("gravitySim <-> potentialSolver", &visSelect, 0, 1);
        if (visSelect == 0) 
        { 
            ImGui::SliderInt("Barnes&Hut <-> naive: ", &gravType, 0, 1);
            if (gravType == 0)
            {
                ImGui::SliderInt("show tree level", &simulation.showLevel, 0, 10);
            }
        }
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
    glfwTerminate();
    return 0;
}



void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, screenWidth, screenHeight);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse) // initially set to true
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        camera.ProcessMouseMovement(xoffset, yoffset);
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}