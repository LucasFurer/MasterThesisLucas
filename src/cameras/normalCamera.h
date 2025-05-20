#pragma once

#include "camera.h"

class NormalCamera : public Camera
{
public:
    NormalCamera
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
        float velocity = MovementSpeed * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            Position += Front * velocity;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            Position -= Front * velocity;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            Position -= Right * velocity;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            Position += Right * velocity;

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            Position += Up * velocity;

        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            Position -= Up * velocity;
    }


    void mouse_callback_impl(GLFWwindow* window, double xposIn, double yposIn) override
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

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        {
            xoffset *= MouseSensitivity;
            yoffset *= MouseSensitivity;

            Yaw += xoffset;
            Pitch += yoffset;

            // make sure that when pitch is out of bounds, screen doesn't get flipped
            if (true)
            {
                if (Pitch > 89.0f)
                    Pitch = 89.0f;
                if (Pitch < -89.0f)
                    Pitch = -89.0f;
            }

            // update Front, Right and Up Vectors using the updated Euler angles
            updateCameraVectors();
        }
    }


    void scroll_callback_impl(GLFWwindow* window, double xoffset, double yoffset) override
    {
        Zoom *= ((-(float)yoffset / 20.0f) + 1.0f);
    }
};