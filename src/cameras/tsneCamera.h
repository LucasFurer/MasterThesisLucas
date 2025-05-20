#pragma once

#include "camera.h"

class TsneCamera : public Camera
{
public:
    TsneCamera
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
    ) : Camera
    (
        initPosition,
        up,
        initYaw,
        initPitch,
        initFront,
        initMovementSpeed,
        initSensitivity,
        initZoom,
        initNearPlane,
        initFarPlane,
        initPerspective,
        initScreenWidth,
        initScreenHeight
    )
    {

    }


    void processInput(GLFWwindow* window, float deltaTime) override
    {
        float velocity = MovementSpeed * Zoom * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            Position += Up * velocity;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            Position -= Up * velocity;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            Position -= Right * velocity;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            Position += Right * velocity;
    }


    void mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn) override
    {

    }


    void scroll_callback_impl(GLFWwindow* window, double xoffset, double yoffset) override
    {
        Zoom *= ((-(float)yoffset / 20.0f) + 1.0f);
    }
};