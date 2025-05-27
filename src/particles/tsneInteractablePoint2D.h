#pragma once

class TsneInteractablePoint2D : public TsneInteractable2D
{
public:
	int label;

    TsneInteractablePoint2D(glm::vec2 initCentreOfMass, int initLabel) : TsneInteractable2D(initCentreOfMass)
    {
        label = initLabel;
    }

    TsneInteractablePoint2D()
    {
        centreOfMass = glm::vec2(0.0f);
        label = 0;
    }



    float getTotalMass() override
    {
        return 1.0f;
    }
    glm::vec2 getCentreOfMass() override
    {
        return centreOfMass;
    }

    glm::vec2 getDipole() override
    {
        return glm::vec2(0.0f);
    }
    Fastor::Tensor<float, 2, 2> getQuadrupole() override
    {
        return Fastor::Tensor<float, 2, 2>{};
    }

    glm::vec2 getHighestCorner() override
    {
        return centreOfMass;
    }
    glm::vec2 getLowestCorner() override
    {
        return centreOfMass;
    }

    std::vector<int>* getOccupants() override
    {
        //std::vector<int> empty;
        //return &empty;
        std::cerr << "this should never happen" << std::endl;
        return nullptr;
    }
    int getOccupantsSize() override
    {
        return 0;
    }

    void addForces(Fastor::Tensor<float, 2> C1add, Fastor::Tensor<float, 2, 2> C2add, Fastor::Tensor<float, 2, 2, 2> C3add) override
    {
        std::cerr << "this should never happen" << std::endl;
    }

    bool areWeNode() override
    {
        return false;
    }
};