#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class EmbeddedPoint
{
public:
    glm::vec2 position;
    int label;

    EmbeddedPoint(glm::vec2 initPosition, int initLabel)
    {
        position = initPosition;
        label = initLabel;
    }

    EmbeddedPoint() : position(glm::vec2(0.0f)), label(0) {}
private:
};