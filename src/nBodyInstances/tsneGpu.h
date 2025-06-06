#pragma once

#include "particles/tsneParticle2D.h"
#include "nbodysolvers/nBodySolverGpuNaive.h"

class TsneGpu
{
public:
    std::vector<int> indexTracker;

    std::vector<TsneParticle2D> tsneParticles;
    std::vector<TsneParticle2D> tsneParticlesPrev;
    std::vector<TsneParticle2D> tsneParticlesPrevPrev;
    //Buffer* TsneParticlesBuffer;

    //Buffer* forceBuffer;
    //float forceSize = 1.0f;

    //std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    //std::map<std::string, NBodySolver<TsneParticle2D>*> nBodySolvers;
    //std::string nBodySelect = "naive";

    NBodySolverGpuNaive<TsneParticle2D> nBodySolverGpuNaive;

    float learnRate;
    float accelerationRate;

    float timeStepsPerSec = 0.0f;
    float lastTimeUpdated = 0.0f;
    int follow = 1;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;



    TsneGpu()
    {
        //srand(time(NULL));
        int TsneParticlesSize = 10000;
        float perplexity = 30.0f;

        learnRate = 1000.0f;
        accelerationRate = 0.5f;


        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/label_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/P_matrix_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif

        labels = Loader::loadLabels(labelsPath.string());
        Pmatrix = Loader::loadPmatrix(fileName.string());

        indexTracker.resize(TsneParticlesSize);
        tsneParticles.resize(TsneParticlesSize);
        tsneParticlesPrev.resize(TsneParticlesSize);
        tsneParticlesPrevPrev.resize(TsneParticlesSize);

        srand(1952732);
        float sizeParam = 2.0f;
        for (int i = 0; i < TsneParticlesSize; i++)
        {
            float randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            float randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;


            while (powf(randX, 2.0f) + powf(randY, 2.0f) > 1.0f)
            {
                randX = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
                randY = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            }


            glm::vec2 pos = glm::vec2(
                powf(sizeParam * randX, 1.0f),
                powf(sizeParam * randY, 1.0f)
            );

            int lab = labels[i];

            indexTracker[i] = i;
            tsneParticles[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab);
            tsneParticlesPrev[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab);
            tsneParticlesPrevPrev[i] = TsneParticle2D(pos, glm::vec2(0.0f), lab);
        }

        //nBodySolvers["naive"] = new NBodySolverNaive<EmbeddedPoint>(&TSNEnaiveKernal);
        //nBodySolvers["BH"] = new NBodySolverBarnesHut<EmbeddedPoint>(&TSNEbarnesHutParticleNodeKernal, &TSNEbarnesHutParticleParticleKernal, 10, 1.0f);
        //nBodySolvers["BH"]->updateTree(&embeddedPoints);
        nBodySolverGpuNaive = NBodySolverGpuNaive<TsneParticle2D>(TSNEGPUnaiveKernal);
        

        //TsneParticlesBuffer = new Buffer(TsneParticles, Pos2floatAcc2floatLab1Int, GL_DYNAMIC_DRAW); // implement Pos2floatAcc2floatLab1Int

        //std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, embeddedDerivative, forceSize);
        //forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
    }

    ~TsneGpu()
    {
        //delete TsneParticlesBuffer;
        //delete forceBuffer;

        //for (std::pair<const std::string, NBodySolver<TsneParticle2D>*> nBodySolverPointer : nBodySolvers)
        //{
        //    delete nBodySolverPointer.second;
        //}
    }

    void cleanup()
    {
        //TsneParticlesBuffer->cleanup();
        //forceBuffer->cleanup();

        //for (std::pair<const std::string, NBodySolver<TsneParticle2D>*> nBodySolver : nBodySolvers)
        //{
        //    nBodySolver.second->boxBuffer->cleanup();
        //}
    }

    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            lastTimeUpdated = glfwGetTime();

            updateDerivative();

            tsneParticlesPrev.swap(tsneParticlesPrevPrev);
            tsneParticles.swap(tsneParticlesPrev);

            for (int i = 0; i < tsneParticles.size(); i++)
            {
                tsneParticles[i].position = tsneParticlesPrev[i].position + learnRate * tsneParticles[i].derivative + accelerationRate * (tsneParticlesPrev[i].position - tsneParticlesPrevPrev[i].position);
            }
        }
    }

private:
    void updateDerivative()
    {
        updateDerivativeRepulsive();

        updateDerivativeAttractive();
    }

    void updateDerivativeRepulsive()
    {
        float accumulator = 0.0f;
        nBodySolverGpuNaive.solveNbody(accumulator, tsneParticles, indexTracker);

        for (int i = 0; i < tsneParticles.size(); i++)
            tsneParticles[indexTracker[i]].derivative *= (1.0f / accumulator);
    }

    void updateDerivativeAttractive()
    {
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                int colIndex = indexTracker[it.col()];
                int rowIndex = indexTracker[it.row()];

                glm::vec2 diff = tsneParticles[colIndex].position - tsneParticles[rowIndex].position;
                float distance = glm::length(diff);

                tsneParticles[colIndex].derivative += -(float)it.value() * (diff / (1.0f + (distance * distance)));
            }
        }
    }
};