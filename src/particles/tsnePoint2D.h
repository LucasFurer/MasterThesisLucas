#pragma once

#include <glm/glm.hpp>
#include <cstdint>

class TsnePoint2D
{
public:
    glm::vec2 position{0.0f};
    glm::vec2 derivative{0.0f};
    int label{0};
    #ifdef INDEX_TRACKER
    int ID{0};
    std::uint32_t morton_code{0};
    #endif

    TsnePoint2D(glm::vec2 initPosition, 
        glm::vec2 initDerivative, 
        int initLabel
        #ifdef INDEX_TRACKER
        , 
        int initID, 
        std::uint32_t init_morton_code
        #endif
    ) : 
        position(initPosition), 
        derivative(initDerivative), 
        label(initLabel)    
        #ifdef INDEX_TRACKER
        ,
        ID(initID), 
        morton_code(init_morton_code)
        #endif
    {}

    TsnePoint2D() {}
};