#pragma once

#include <glm/glm.hpp>

class TsnePoint2D
{
public:
    glm::vec2 position;
    glm::vec2 derivative;
    int label;
    int ID;

    TsnePoint2D(glm::vec2 initPosition, glm::vec2 initDerivative, int initLabel, int initID)
    {
        position = initPosition;
        derivative = initDerivative;
        label = initLabel;
        ID = initID;
    }

    TsnePoint2D()
    {
        position = glm::vec2(0.0f);
        derivative = glm::vec2(0.0f);
        label = 0;
        ID = 0;
    }
private:
};