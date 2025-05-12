#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement 
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};


class Camera
{
public:
    // camera Attributes
    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // camera options
    float MovementSpeed;
    float MouseSensitivity;
    float Zoom;
    float nearPlane;
    float farPlane;
    bool perspective;

    unsigned int* screenWidth;
    unsigned int* screenHeight;

    float lastX = 400;
    float lastY = 300;
    bool firstMouse = true;



    Camera
    (
        glm::vec3 initPosition,
        glm::vec3 up,
        float initYaw,
        float initPitch,
        glm::vec3 initFront,
        float initMovementSpeed,
        float initSensitivity,
        float initZoom,
        float initNearPlane,
        float initFarPlane,
        bool initPerspective,
        unsigned int* initScreenWidth,
        unsigned int* initScreenHeight
    )
    {
        Position = initPosition;
        Front = initFront;
        WorldUp = up;

        Yaw = initYaw;
        Pitch = initPitch;

        MovementSpeed = initMovementSpeed;
        MouseSensitivity = initSensitivity;
        Zoom = initZoom;
        nearPlane = initNearPlane;
        farPlane = initFarPlane;
        perspective = initPerspective;

        screenWidth = initScreenWidth;
        screenHeight = initScreenHeight;

        updateCameraVectors();
    }

    // returns the view matrix calculated using Euler Angles and the LookAt Matrix
    glm::mat4 getViewMatrix()
    {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 getProjectionMatrix()
    {
        if (perspective) 
        { 
            if (Zoom < 1.0f)
                Zoom = 1.0f;
            if (Zoom > 45.0f)
                Zoom = 45.0f;
            return glm::perspective(glm::radians(Zoom), (float)*screenWidth / (float)*screenHeight, nearPlane, farPlane);
        }
        else 
        { 
            /*
            
            float halfHeight = tan(glm::radians(Zoom) / 2.0f) * nearPlane; // Half height at nearPlane
            float halfWidth = halfHeight * aspect;

            std::cout << halfWidth << std::endl;
            std::cout << halfHeight << std::endl;

            return glm::ortho(
                -halfWidth, halfWidth,   // left, right
                -halfHeight, halfHeight, // bottom, top
                nearPlane, farPlane      // near, far
            );
            */
            float aspect = (float)*screenWidth / (float)*screenHeight;
            //return glm::ortho(-200.0f * aspect, 200.0f * aspect, -200.0f, 200.0f, nearPlane, farPlane);
            return glm::ortho(-Zoom * 1.0f * aspect, Zoom * 1.0f * aspect, -Zoom * 1.0f, Zoom * 1.0f, nearPlane, farPlane);
        }
    }


    virtual void processInput(GLFWwindow* window, float deltaTime) = 0;

    static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
    {
        ImGui_ImplGlfw_CursorPosCallback(window, xposIn, yposIn); // needed so that imgui stays responsive
        Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
        cam->mouse_callback_impl(window, xposIn, yposIn);
    }
    virtual void mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn) = 0;

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
        cam->scroll_callback_impl(window, xoffset, yoffset);
    }
    virtual void scroll_callback_impl(GLFWwindow* window, double xoffset, double yoffset) = 0;


    // calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        // also re-calculate the Right and Up vector
        Right = glm::normalize(glm::cross(Front, WorldUp));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
        Up = glm::normalize(glm::cross(Right, Front));
    }
private:
};
