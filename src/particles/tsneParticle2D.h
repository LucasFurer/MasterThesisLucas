#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class TsneParticle2D
{
public:
    glm::vec2 position;
    glm::vec2 derivative;
    int label;

    TsneParticle2D(glm::vec2 initPosition, glm::vec2 initDerivative, int initLabel)
    {
        position = initPosition;
        derivative = initDerivative;
        label = initLabel;
    }

    TsneParticle2D()
    {
        position = glm::vec2(0.0f);
        label = 0;
    }
private:
};