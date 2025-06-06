#pragma once

float verticesFlat[] = {
    // positions          // colors           // texture coords
     0.5f,  0.5f, 0.5f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
     0.5f, -0.5f, 0.5f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -0.5f, -0.5f, 0.5f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
    -0.5f,  0.5f, 0.5f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left 
};
unsigned int indicesFlat[] = {  // note that we start from 0!
    0, 1, 3,   // first triangle
    1, 2, 3    // second triangle
};



float verticesCube[] = {
    // positions          //normals                                                     // colors           // texture coords
    -1.0f, -1.0f, -1.0f,  -1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
     1.0f, -1.0f, -1.0f,   1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
     1.0f,  1.0f, -1.0f,   1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
    -1.0f,  1.0f, -1.0f,  -1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   0.0f, 1.0f,
    -1.0f, -1.0f,  1.0f,  -1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   1.0f, 1.0f,
     1.0f, -1.0f,  1.0f,   1.0f / sqrt(3.0f), -1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   1.0f, 0.0f,
     1.0f,  1.0f,  1.0f,   1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   0.0f, 0.0f,
    -1.0f,  1.0f,  1.0f,  -1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),  1.0f / sqrt(3.0f),   1.0f, 1.0f, 1.0f,   0.0f, 1.0f
};
unsigned int indicesCube[] = {  // note that we start from 0!
    0, 1, 3, 
    3, 1, 2,

    1, 5, 2, 
    2, 5, 6,

    5, 4, 6, 
    6, 4, 7,

    4, 0, 7, 
    7, 0, 3,

    3, 2, 7, 
    7, 2, 6,

    4, 5, 0, 
    0, 5, 1
};


float verticesQuad[] = {
    // positions          // texture coords
     1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
};
unsigned int indicesQuad[] = {  // note that we start from 0!
    0, 1, 3,   // first triangle
    1, 2, 3    // second triangle
};