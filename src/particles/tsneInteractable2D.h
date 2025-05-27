#pragma once

class TsneInteractable2D
{
public:
    glm::vec2 centreOfMass;

    TsneInteractable2D(glm::vec2 initCentreOfMass)
    {
        centreOfMass = initCentreOfMass;
    }

    TsneInteractable2D() 
    {
        centreOfMass = glm::vec2(0.0f);
    }


    virtual float getTotalMass() = 0;
    virtual glm::vec2 getCentreOfMass() = 0;

    virtual glm::vec2 getDipole() = 0;
    virtual Fastor::Tensor<float, 2, 2> getQuadrupole() = 0;

    virtual glm::vec2 getHighestCorner() = 0;
    virtual glm::vec2 getLowestCorner() = 0;

    virtual std::vector<int>* getOccupants() = 0;
    virtual int getOccupantsSize() = 0;

    virtual void addForces(Fastor::Tensor<float, 2> C1add, Fastor::Tensor<float, 2, 2> C2add, Fastor::Tensor<float, 2, 2, 2> C3add) = 0;

    virtual bool areWeNode() = 0;
};