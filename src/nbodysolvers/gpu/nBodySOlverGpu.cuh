#pragma once

#include <vector>

#include "../../particles/tsneParticle2D.h"
#include "../../structs/sparseEntry2D.h"
#include "../../common.h"
#include "../../openGLhelper/buffer.h"

template <typename T>
class NBodySolverGpu
{
public:
    std::vector<LineSegment2D> lineSegments;
    Buffer* boxBuffer = new Buffer(); // wait maybe dont call new buffer because an empty buffer will still create its own buffers with opengl
    int showLevel = 0;

    virtual ~NBodySolverGpu() 
    {
        delete boxBuffer;
    }

    void cleanup()
    {
        boxBuffer->cleanup();
    }

    virtual void getParticles(std::vector<TsneParticle2D>& result) = 0;

    virtual void timeStep() = 0;

    virtual void getTree() = 0;
};