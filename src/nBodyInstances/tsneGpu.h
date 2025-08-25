#pragma once

#include "particles/tsneParticle2D.h"
//#include "nbodysolvers/gpu/nBodySolverGpuNaive.h"
//#include "nbodysolvers/gpu/nBodySolverGpuBH.h"
#include "../../nbodysolvers/gpu/nBodySolverGpuNaive.cuh"
#include "../../structs/sparseEntry2D.h"





class TsneGpu
{
public:
    //std::vector<int> indexTracker;
    //std::vector<int> indexTrackerPrev;

    std::vector<TsneParticle2D> tsneParticlesToShow;
    //std::vector<TsneParticle2D> tsneParticles;
    //std::vector<TsneParticle2D> tsneParticlesPrev;
    //std::vector<TsneParticle2D> tsneParticlesPrevPrev;
    Buffer* TsneParticlesBuffer; // essential

    Buffer* forceBuffer; // essential
    float forceSize = 1.0f;

    //std::vector<LineSegment2D> lineSegments;
    //Buffer* boxBuffer = new Buffer();
    //int showLevel = 0;

    std::map<std::string, NBodySolverGpuNaive<TsneParticle2D>*> nBodySolvers; // std::map<std::string, NBodySolverGpu<TsneParticle2D>*>
    std::string nBodySelect = "naive";

    //float learnRate;
    //float accelerationRate;

    float timeStepsPerSec = 0.0f;
    float lastTimeUpdated = 0.0f;
    int follow = 1;

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;



    TsneGpu()
    {
        // set parameters for t-SNE data input
        int TsneParticlesSize = 10000;
        float perplexity = 30.0f;


        // get the path of the labels and sparse matrix
        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/label_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/P_matrix_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(TsneParticlesSize) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif

        // load the labels and sparse matrix
        labels = Loader::loadLabels(labelsPath.string());
        int* labelsArr = new int[TsneParticlesSize];
        for (int i = 0; i < TsneParticlesSize; i++)
        {
            labelsArr[i] = labels[i];
        }

        Pmatrix = Loader::loadPmatrix(fileName.string());
        std::vector<SparseEntryCOO2D> sparseMatrixCOO = std::vector<SparseEntryCOO2D>(Pmatrix.nonZeros());
        int indexSparse = 0;
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                //std::cout << "col: " << it.col() << ", row: " << it.row() << ", value: " << it.value() << std::endl;
                sparseMatrixCOO[indexSparse] = SparseEntryCOO2D(it.col(), it.row(), it.value());
                indexSparse++;
            }
        }

        // sort the sparse matrix entries so we can put it into a CSC structure
        std::sort(sparseMatrixCOO.begin(), sparseMatrixCOO.end(),
            [](const SparseEntryCOO2D& a, const SparseEntryCOO2D& b) 
            {
                if (a.col == b.col) 
                {
                    return a.row < b.row;
                }
                return a.col < b.col;
            }
        );

        // put the COO entries in a CSC format
        SparseEntryCSC2D* sparseMatrixCSC = new SparseEntryCSC2D[sparseMatrixCOO.size()];
        for (int i = 0; i < sparseMatrixCOO.size(); i++)
        {
            sparseMatrixCSC[i] = SparseEntryCSC2D(sparseMatrixCOO[i].row, sparseMatrixCOO[i].val);
        }

        // since we use the CSC format we lose the col data and thus we use this array to keep track
        int* sparseMatrixColumnIndexStart = new int[TsneParticlesSize + 1];
        sparseMatrixColumnIndexStart[0] = 0;

        int sparseMatrixCOOIndex = 0;
        for (int c = 0; c < TsneParticlesSize; c++)
        {
            while (sparseMatrixCOOIndex < sparseMatrixCOO.size() && sparseMatrixCOO[sparseMatrixCOOIndex].col == c)
            {
                sparseMatrixCOOIndex++;
            }
            sparseMatrixColumnIndexStart[c + 1] = sparseMatrixCOOIndex;
        }

        tsneParticlesToShow.resize(TsneParticlesSize);
        //indexTracker.resize(TsneParticlesSize);
        //indexTrackerPrev.resize(TsneParticlesSize);

        //tsneParticles.resize(TsneParticlesSize);
        //tsneParticlesPrev.resize(TsneParticlesSize);
        //tsneParticlesPrevPrev.resize(TsneParticlesSize);

        //srand(time(NULL));

        //sparseMatrixCSC with size sparseMatrixCOO.size()
        //sparseMatrixColumnIndexStart with size TsneParticlesSize + 1

        nBodySolvers["naive"] = new NBodySolverGpuNaive<TsneParticle2D>(TsneParticlesSize, sparseMatrixCSC, sparseMatrixCOO.size(), sparseMatrixColumnIndexStart, labelsArr, 1000.0f, 0.2f); // TSNEGPUnaiveKernal
        //nBodySolvers["BH"] = new NBodySolverBarnesHut<EmbeddedPoint>(&TSNEbarnesHutParticleNodeKernal, &TSNEbarnesHutParticleParticleKernal, 10, 1.0f);
        //nBodySolvers["BH"]->updateTree(&embeddedPoints);


        nBodySolvers[nBodySelect]->getParticles(tsneParticlesToShow);


        TsneParticlesBuffer = new Buffer(Pos2FloatLab1Int::particlesToVertexPos2Col3(tsneParticlesToShow), pos2DlabelInt, GL_DYNAMIC_DRAW); // implement Pos2floatAcc2floatLab1Int
        forceBuffer = new Buffer(VertexPos2Col3::particlesToVertexPos2Col3(tsneParticlesToShow, forceSize), pos2DCol3D, GL_DYNAMIC_DRAW);
    }

    ~TsneGpu()
    {
        delete TsneParticlesBuffer;
        delete forceBuffer;

        for (std::pair<const std::string, NBodySolverGpuNaive<TsneParticle2D>*> nBodySolverPointer : nBodySolvers) // std::pair<const std::string, NBodySolverGpu<TsneParticle2D>*>
        {
            delete nBodySolverPointer.second;
        }
    }

    void cleanup()
    {
        TsneParticlesBuffer->cleanup();
        forceBuffer->cleanup();

        for (std::pair<const std::string, NBodySolverGpuNaive<TsneParticle2D>*> nBodySolver : nBodySolvers)
        {
            //nBodySolver.second->cleanup();
        }
    }

    void timeStep()
    {
        if (glfwGetTime() - lastTimeUpdated >= 1.0f / timeStepsPerSec)
        {
            //std::cout << "test int: " << nBodySolvers["naive"]->testT.position.x << std::endl;

            lastTimeUpdated = glfwGetTime();

            //updateDerivative();

            //tsneParticlesPrev.swap(tsneParticlesPrevPrev);
            //tsneParticles.swap(tsneParticlesPrev);

            //for (int i = 0; i < tsneParticles.size(); i++)
            //{
            //    tsneParticles[indexTracker[i]].position = tsneParticlesPrev[indexTracker[i]].position + learnRate * tsneParticlesPrev[indexTracker[i]].derivative + accelerationRate * (tsneParticlesPrev[indexTracker[i]].position - tsneParticlesPrevPrev[indexTrackerPrev[i]].position);
            //    tsneParticles[indexTracker[i]].label = tsneParticlesPrev[indexTracker[i]].label;
            //    tsneParticles[indexTracker[i]].ID = tsneParticlesPrev[indexTracker[i]].ID;
            //}
            //indexTrackerPrev = indexTracker;

            nBodySolvers[nBodySelect]->timeStep();

            nBodySolvers[nBodySelect]->getParticles(tsneParticlesToShow);

            TsneParticlesBuffer->updateBuffer(Pos2FloatLab1Int::particlesToVertexPos2Col3(tsneParticlesToShow), pos2DlabelInt);
            //nBodySolvers[nBodySelect]->updateTree(tsneParticles, indexTracker);
            forceBuffer->updateBuffer(VertexPos2Col3::particlesToVertexPos2Col3(tsneParticlesToShow, forceSize), pos2DCol3D);
        }
    }

    /*
    void tests()
    {
        int maxIterations = 2;

        float totalError = 0.0f;
        for (int i = 0; i < maxIterations; i++)
        {
            std::vector<TsneParticle2D> tsneParticlesOther = tsneParticles;
            //std::vector<int> indexTrackerOther = indexTracker;

            float accumulator = 0.0f;
            float accumulatorOther = 0.0f;
            //nBodySolvers["naive"]->solveNbody(accumulator, tsneParticles, indexTracker);
            //nBodySolvers["naive"]->solveNbody(accumulatorOther, tsneParticlesOther, indexTracker); // other thing

            totalError += getMSE(tsneParticles, tsneParticlesOther);

            for (int i = 0; i < tsneParticles.size(); i++)
                tsneParticles[i].derivative *= (1.0f / accumulator);

            updateDerivativeAttractive();

            tsneParticlesPrev.swap(tsneParticlesPrevPrev);
            tsneParticles.swap(tsneParticlesPrev);

            for (int i = 0; i < tsneParticles.size(); i++)
            {
                tsneParticles[i].position = tsneParticlesPrev[i].position + learnRate * tsneParticles[i].derivative + accelerationRate * (tsneParticlesPrev[i].position - tsneParticlesPrevPrev[i].position);
            }
        }
        float averageError = totalError / static_cast<float>(maxIterations);

        std::cout << "average error: " << averageError << std::endl;
    }
    */

    std::tuple<float, float, float, float> getEdges()
    {
        float left = std::numeric_limits<float>::max();
        float down = std::numeric_limits<float>::max();
        float right = std::numeric_limits<float>::min();
        float up = std::numeric_limits<float>::min();

        for (int i = 0; i < tsneParticlesToShow.size(); i++)
        {
            float posX = tsneParticlesToShow[i].position.x;
            float posY = tsneParticlesToShow[i].position.y;
            
            left  = std::min(posX, left);
            right = std::max(posX, right);
            down  = std::min(posY, down);
            up    = std::max(posY, up);
        }

        return std::make_tuple(left, right, down, up);
    }

private:
    /*
    void updateDerivative()
    {
        updateDerivativeRepulsive();

        updateDerivativeAttractive();
    }

    void updateDerivativeRepulsive()
    {
        float accumulator = 0.0f;
        //nBodySolvers[nBodySelect]->solveNbody(accumulator, tsneParticles, indexTracker);

        for (TsneParticle2D& tsneParticle2D : tsneParticles)
            tsneParticle2D.derivative *= -(1.0f / accumulator);
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

    float getMSE(const std::vector<TsneParticle2D>& tsneParticles, const std::vector<TsneParticle2D>& tsneParticlesOther)
    {
        float MSE = 0.0f;
        float divide = 0.0f;

        for (int i = 0; i < tsneParticles.size(); i++)
        {
            MSE += powf(glm::length(tsneParticles[tsneParticles[i].ID].derivative - tsneParticlesOther[tsneParticlesOther[i].ID].derivative), 1.0f);
            divide += powf(glm::length(tsneParticles[tsneParticles[i].ID].derivative), 1.0f);
        }

        float NMSE = MSE / divide;
        return NMSE;
    }
    */
};