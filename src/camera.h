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

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
    void processKeyboard(int key, float deltaTime)
    {
        float velocity = MovementSpeed * deltaTime;
        if (key == GLFW_KEY_W)
            Position += Front * velocity;
        if (key == GLFW_KEY_S)
            Position -= Front * velocity;
        if (key == GLFW_KEY_A)
            Position -= Right * velocity;
        if (key == GLFW_KEY_D)
            Position += Right * velocity;
        if (key == GLFW_KEY_LEFT_SHIFT)
            Position += Up * velocity;
        if (key == GLFW_KEY_LEFT_CONTROL)
            Position -= Up * velocity;
    }

    // processes input received from a mouse input system. Expects the offset value in both the x and y direction.
    void processMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
    {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        // make sure that when pitch is out of bounds, screen doesn't get flipped
        if (constrainPitch)
        {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        // update Front, Right and Up Vectors using the updated Euler angles
        updateCameraVectors();
    }

    // processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
    void processMouseScroll(float yoffset)
    {
        Zoom -= (float)yoffset;
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
