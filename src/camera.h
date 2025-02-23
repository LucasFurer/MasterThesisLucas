#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Default camera values
//const float YAW = 90.0f;
//const float PITCH = 0.0f;
//const float SPEED = 12.5f;
//const float SENSITIVITY = 0.1f;
//const float ZOOM = 45.0f;



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


    void processInput(GLFWwindow* window, float deltaTime)
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

    static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
    {
        Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));

        float xpos = static_cast<float>(xposIn);
        float ypos = static_cast<float>(yposIn);

        if (cam->firstMouse) // initially set to true
        {
            cam->lastX = xpos;
            cam->lastY = ypos;
            cam->firstMouse = false;
        }

        float xoffset = xpos - cam->lastX;
        float yoffset = cam->lastY - ypos; // reversed since y-coordinates range from bottom to top
        cam->lastX = xpos;
        cam->lastY = ypos;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) 
        {
            xoffset *= cam->MouseSensitivity;
            yoffset *= cam->MouseSensitivity;

            cam->Yaw += xoffset;
            cam->Pitch += yoffset;

            // make sure that when pitch is out of bounds, screen doesn't get flipped
            if (true)
            {
                if (cam->Pitch > 89.0f)
                    cam->Pitch = 89.0f;
                if (cam->Pitch < -89.0f)
                    cam->Pitch = -89.0f;
            }

            // update Front, Right and Up Vectors using the updated Euler angles
            cam->updateCameraVectors();
        }
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));

        //cam->Zoom -= (float)yoffset;
        cam->Zoom *= ((-(float)yoffset / 20.0f) + 1.0f);
        //if (Zoom < 1.0f)
        //    Zoom = 1.0f;
        //if (Zoom > 45.0f)
        //    Zoom = 45.0f;
    }

private:
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
};
#endif
