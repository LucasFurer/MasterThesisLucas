#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <filesystem>
#include <string>
#include <utility>
#include <unsupported/Eigen/SparseExtra>
#include <limits>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/string_cast.hpp>
#include <exception>
#include <GLFW/glfw3.h>
#include <cstddef>
#include <numbers>

#include "../particles/embeddedPoint.h"
#include "../particles/tsnePoint2D.h"
#include "../openGLhelper/buffer.h"
#include "../common.h"
#include "../openGLhelper/buffer.h"
#include "../dataLoaders/loader.h"
#include "../nbodysolvers/cpu/nBodySolverNaive.h"
#include "../nbodysolvers/cpu/nBodySolverBH.h"
#include "../nbodysolvers/cpu/nBodySolverBHR.h"
#include "../nbodysolvers/cpu/nBodySolverBHRMP.h"
#include "../nbodysolvers/cpu/nBodySolverBHMP.h"
#include "../nbodysolvers/cpu/nBodySolverFMM.h"
#include "../nbodysolvers/cpu/nBodySolverFMM_MORTON.h"
#include "../nbodysolvers/cpu/nBodySolverFMM_SYM_MORTON.h"
#include "../nbodysolvers/cpu/nBodySolverPM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
#include "../ffthelper.h"
#include "../Timer.h"
#include "../particleMesh.h"

//#define TIME_ALGORITHM


class TSNE
{
public:
    std::vector<TsnePoint2D> embeddedPoints;
    std::vector<TsnePoint2D> embeddedPointsPrev;
    std::vector<TsnePoint2D> embeddedPointsPrevPrev;
    Buffer* embeddedBuffer;

    #ifdef INDEX_TRACKER
    std::vector<int> indexTracker;
    std::vector<int> indexTrackerPrev;
    #endif

    int follow = 1;
    glm::vec2 minPos;
    glm::vec2 maxPos;

    int nodeLevelToShow = 0;
    Buffer* nodeBuffer;

    float forceSize = 1.0f;
    Buffer* forceBuffer;

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect = "naive"; // default selector

    float learnRate;
    float accelerationRate; // kinda irrelevant since this is based on iteration_counter

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    std::vector<unsigned int> row_P;
    std::vector<unsigned int> col_P;
    std::vector<double> val_P;

    float desired_iteration_per_second; // limits the speed of tsne
    float time_since_last_iteration;

    int iteration_counter = 0; // keeps track of which iteration we are on

    Timer thousand_iteration_timer; 
    bool reached_thousand_iterations = false;

    //float min_theta = 0.5f;
    float min_theta = 0.75f;
    //float max_theta = 2.0f;
    float max_theta = 2.0f;

    float cell_size = 3.5f;
    
	TSNE()
	{
        int dataAmount = 70000;
        float perplexity = 30.0f;
        std::string dataSet = "MNIST_digits"; // "MNIST_digits", "MNIST_fashion", "mice_brain_cells", "CIFAR10"

        learnRate = static_cast<float>(dataAmount) / 15.0f;

        desired_iteration_per_second = 0.0f;

        time_since_last_iteration = 0.0f;

        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/" + dataSet + "/" + std::to_string(dataAmount) + "/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/" + dataSet + "/" + std::to_string(dataAmount) + "/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(dataAmount) + "_perp" + std::to_string((int)perplexity) + ".mtx");
        #endif
        labels = Loader::loadLabels(labelsPath.string());
        Pmatrix = Loader::loadPmatrix(fileName.string());

        embeddedPoints.resize(dataAmount);
        embeddedPointsPrev.resize(dataAmount);
        embeddedPointsPrevPrev.resize(dataAmount);
        
        #ifdef INDEX_TRACKER
        indexTracker.resize(dataAmount);
        indexTrackerPrev.resize(dataAmount);
        #endif




        #ifdef TIME_ALGORITHM 
        Timer time_point_creation;
        #endif
        
        //srand(time(NULL));
        srand(296343u);
        float sizeParam = 2.0f;
        for (int i = 0; i < dataAmount; i++)
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
            
            #ifdef INDEX_TRACKER
            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            #else
            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            #endif 

            #ifdef INDEX_TRACKER
            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
            #endif
        }
        #ifdef TIME_ALGORITHM 
        time_point_creation.endTimer("tsne point creation");
        #endif

        setupEfficientPmatrix();

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("tsne update minmax");
        #endif

        float set_theta = 1.0f;
        float set_cell_size = cell_size;
        int max_children_per_node = 16;
        nBodySolvers["naive"] = new NBodySolverNaive<TsnePoint2D>(&TSNEnaiveKernel);
        nBodySolvers["BH"] = new NBodySolverBH<TsnePoint2D>(&TSNEBHPNKernel, &TSNEBHPPKernel, max_children_per_node, set_theta);
        nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHR"] = new NBodySolverBHR<TsnePoint2D>(&TSNEBHRNPKernel, &TSNEBHRPPKernel, max_children_per_node, set_theta);
        nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHMP"] = new NBodySolverBHMP<TsnePoint2D>(&TSNEBHMPPNKernel, &TSNEBHMPPPKernel, max_children_per_node, set_theta);
        nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHRMP"] = new NBodySolverBHRMP<TsnePoint2D>(&TSNEBHRMPNPKernel, &TSNEBHRMPPPKernel, max_children_per_node, set_theta);
        nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM"] = new NBodySolverFMM<TsnePoint2D>(&TSNEFMMNNKernel, &TSNEFMMPNKernel, &TSNEFMMNPKernel, &TSNEFMMPPKernel, max_children_per_node, set_theta);
        nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["PM"] = new NBodySolverPM<TsnePoint2D>(Pmatrix, embeddedPoints, 4, set_cell_size, 4);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
        #ifdef INDEX_TRACKER
        nBodySolvers["FMM_MORTON"] = new NBodySolverFMM_MORTON<TsnePoint2D>(&TSNEFMM_MORTONNNKernel, &TSNEFMM_MORTONPNKernel, &TSNEFMM_MORTONNPKernel, &TSNEFMM_MORTONPPKernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7f, dataAmount), set_theta);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"] = new NBodySolverFMM_SYM_MORTON<TsnePoint2D>(&TSNE_FMM_SYM_MORTON_NN_Kernel, &TSNE_FMM_SYM_MORTON_PN_Kernel, &TSNE_FMM_SYM_MORTON_PP_Kernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7f, dataAmount), set_theta);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        #endif

        #ifdef INDEX_TRACKER
        embeddedBuffer = new Buffer(embeddedPoints, Float2Float2Int1Int1Int32_t1, GL_DYNAMIC_DRAW);
        #else
        embeddedBuffer = new Buffer(embeddedPoints, Float2Float2Int1, GL_DYNAMIC_DRAW);
        #endif

        std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
        nodeBuffer = new Buffer(nodesBufferData, pos2DCol3D, GL_DYNAMIC_DRAW);
        
        std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, forceSize);
        forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
	}
	
	~TSNE()
	{
        delete embeddedBuffer;
        delete nodeBuffer;
        delete forceBuffer;

        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }

        nBodySolvers.clear();
	}

    void cleanup()
    {
        embeddedBuffer->cleanup();
        nodeBuffer->cleanup();
        forceBuffer->cleanup();
    }

    void resetTsne(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, float theta, float cell_size, unsigned int seed)
    {
        learnRate = static_cast<float>(data_size) / 15.0f;
        desired_iteration_per_second = 0.0f;
        time_since_last_iteration = 0.0f;

        #ifdef _WIN32
        std::filesystem::path labelsPath = std::filesystem::current_path() / ("data/" + dataset_type + "/" + std::to_string(data_size) + "/label_amount" + std::to_string(data_size) + "_perp" + std::to_string((int)perplexity_value) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path() / ("data/" + dataset_type + "/" + std::to_string(data_size) + "/P_matrix_amount" + std::to_string(data_size) + "_perp" + std::to_string((int)perplexity_value) + ".mtx");
        #endif
        #ifdef linux
        std::filesystem::path labelsPath = std::filesystem::current_path().parent_path() / ("data/label_amount" + std::to_string(data_size) + "_perp" + std::to_string((int)perplexity_value) + ".bin");
        std::filesystem::path fileName = std::filesystem::current_path().parent_path() / ("data/P_matrix_amount" + std::to_string(data_size) + "_perp" + std::to_string((int)perplexity_value) + ".mtx");
        #endif

        labels = Loader::loadLabels(labelsPath.string());
        Pmatrix = Loader::loadPmatrix(fileName.string());

        embeddedPoints.resize(data_size);
        embeddedPointsPrev.resize(data_size);
        embeddedPointsPrevPrev.resize(data_size);
        
        #ifdef INDEX_TRACKER
        indexTracker.resize(data_size);
        indexTrackerPrev.resize(data_size);
        #endif

        #ifdef TIME_ALGORITHM 
        Timer time_point_creation;
        #endif
        
        //srand(time(NULL));
        srand(seed);
        float sizeParam = 2.0f;
        for (int i = 0; i < data_size; i++)
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
            
            #ifdef INDEX_TRACKER
            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i, 0.0);
            #else
            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab);
            #endif      

            #ifdef INDEX_TRACKER
            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
            #endif
        }
        #ifdef TIME_ALGORITHM 
        time_point_creation.endTimer("tsne point creation");
        #endif

        setupEfficientPmatrix();

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("tsne update minmax");
        #endif

        setThetaForAll(theta, cell_size);
        nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
        #ifdef INDEX_TRACKER
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        #endif


        #ifdef INDEX_TRACKER
        embeddedBuffer = new Buffer(embeddedPoints, Float2Float2Int1Int1Int32_t1, GL_DYNAMIC_DRAW);
        #else
        embeddedBuffer = new Buffer(embeddedPoints, Float2Float2Int1, GL_DYNAMIC_DRAW);
        #endif

        std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
        nodeBuffer = new Buffer(nodesBufferData, pos2DCol3D, GL_DYNAMIC_DRAW);
        
        std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, forceSize);
        forceBuffer = new Buffer(forceLines, pos2DCol3D, GL_DYNAMIC_DRAW);
    }
    
    void timeStep()
    {
        if (iteration_counter == 0)
            thousand_iteration_timer.startTimer();

        if (iteration_counter == 1000 && !reached_thousand_iterations)
        {
            reached_thousand_iterations = true;
            desired_iteration_per_second = 0.0f;
            thousand_iteration_timer.endTimer("A thousand iterations");
        }
        

        if (glfwGetTime() - time_since_last_iteration >= 1.0f / desired_iteration_per_second)
        {
            Timer time_step_timer;

            time_since_last_iteration = glfwGetTime();

            Timer derivative_timer;
            updateDerivative();
            derivative_timer.endTimer("updateDerivative");

            Timer update_timer;
            updatePoints();
            update_timer.endTimer("updatePoints");

            #ifdef INDEX_TRACKER
            embeddedBuffer->updateBuffer(embeddedPoints, Float2Float2Int1Int1Int32_t1);
            #else
            embeddedBuffer->updateBuffer(embeddedPoints, Float2Float2Int1);
            #endif


            Timer time_update_tree;
            nBodySolvers[nBodySelect]->updateTree(embeddedPoints, minPos, maxPos);
            std::vector<VertexPos2Col3> nodesBufferData = nBodySolvers[nBodySelect]->getNodesBufferData(nodeLevelToShow);
            nodeBuffer->updateBuffer(nodesBufferData, pos2DCol3D);
            time_update_tree.endTimer("update tree");


            std::vector<VertexPos2Col3> forceLines = VertexPos2Col3::particlesAccelerationsToVertexPos2Col3(embeddedPoints, forceSize);
            forceBuffer->updateBuffer(forceLines, pos2DCol3D);
            

            time_step_timer.endTimer("timeStep");
        }
    }

    void updatePoints()
    {
        //Timer timer;

        //if (iteration_counter == 999)
        //    costFunction(embeddedPoints, indexTracker, Pmatrix);

        embeddedPointsPrev.swap(embeddedPointsPrevPrev);
        embeddedPoints.swap(embeddedPointsPrev);

        thetaFunction();

        accelerationRate = 0.8f;
        if (iteration_counter < 250)
        {
            accelerationRate = 0.5f;
        }

        #ifdef TIME_ALGORITHM 
        Timer time_update;
        #endif
        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            #ifdef INDEX_TRACKER
            int indexPrev = indexTracker[i];
            int indexPrevPrev = indexTrackerPrev[i];

            embeddedPoints[indexPrev].derivative = embeddedPointsPrev[indexPrev].derivative;
            embeddedPoints[indexPrev].label = embeddedPointsPrev[indexPrev].label;
            embeddedPoints[indexPrev].ID = embeddedPointsPrev[indexPrev].ID;
            embeddedPoints[indexPrev].position = embeddedPointsPrev[indexPrev].position + learnRate * embeddedPointsPrev[indexPrev].derivative + accelerationRate * (embeddedPointsPrev[indexPrev].position - embeddedPointsPrevPrev[indexPrevPrev].position);
            #else   
            embeddedPoints[i].derivative = embeddedPointsPrev[i].derivative;
            embeddedPoints[i].label = embeddedPointsPrev[i].label;
            embeddedPoints[i].position = embeddedPointsPrev[i].position + learnRate * embeddedPointsPrev[i].derivative + accelerationRate * (embeddedPointsPrev[i].position - embeddedPointsPrevPrev[i].position);
            #endif
        }
        #ifdef TIME_ALGORITHM 
        time_update.endTimer("update positions");
        #endif

        #ifdef INDEX_TRACKER
        indexTrackerPrev.swap(indexTracker);
        #endif

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("update minmax");
        #endif

        iteration_counter++;

        //timer.endTimer("update step");
    }

    void thetaFunction()
    {
        Timer timer;

        //float falloff_strength = 5.0f;
        //float theta_result = 
        //    (max_theta - min_theta) *
        //    std::max(1.0f - std::pow(static_cast<float>(iteration_counter) / 1000.0f, falloff_strength), 0.0f) +
        //    min_theta;

        float sharpness = 2.0f;
        float extra_theta_ratio = 0.5f + 0.5f * std::cos
        (
            std::numbers::pi_v<float> *
            std::min
            (
                std::pow(static_cast<float>(iteration_counter) / 999.0f, sharpness),
                1.0f
            )
        );
        float theta_diff = max_theta - min_theta;
        float theta_result = min_theta + extra_theta_ratio * theta_diff;

        std::cout << "set theta to: " << theta_result << std::endl;
        std::cout << "set cell size to: " << cell_size << std::endl;

        setThetaForAll(theta_result, cell_size);

        timer.endTimer("calculating theta");
    }

    void setMinMaxTheta(float set_min_theta, float set_max_theta)
    {
        min_theta = set_min_theta;
        max_theta = set_max_theta;
    }

    void setThetaForAll(float new_theta, float new_cell_size)
    {
        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            nBodySolverPointer.second->theta = new_theta;
            nBodySolverPointer.second->cell_size = new_cell_size;
        }
    }

    void updateAllTrees()
    {
        updateMinMaxPos();

        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            nBodySolverPointer.second->updateTree(embeddedPoints, minPos, maxPos);
        }
    }

    void updateMinMaxPos()
    {
        minPos = glm::vec2(std::numeric_limits<float>::max());
        maxPos = glm::vec2(std::numeric_limits<float>::lowest());

        for (TsnePoint2D& point : embeddedPoints)
        {
            minPos = glm::min(minPos, point.position);
            maxPos = glm::max(maxPos, point.position);
        }
    }

    void updateDerivative()
    {
        #ifdef INDEX_TRACKER
        updateIndexTracker();
        #endif

        resetDeriv();

        updateRepulsive();

        updateAttractive();
    }

    #ifdef INDEX_TRACKER
    void updateIndexTracker()
    {
        for (int i = 0; i < embeddedPoints.size(); i++)
            indexTracker[embeddedPoints[i].ID] = i;
    }
    #endif

    void resetDeriv()
    {
        for (TsnePoint2D& embeddedPoint : embeddedPoints)
            embeddedPoint.derivative = glm::vec2(0.0f);
    }

    void updateRepulsive()
    {
        Timer timer;

        double QijTotal = 0.0f;

        #ifdef INDEX_TRACKER
        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints, indexTracker);
        #else
        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints);
        #endif

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedPoints[i].derivative *= (4.0 / QijTotal);
        }

        timer.endTimer("repulsive");
    }

    void updateAttractiveOUTDATED()
    {
        Timer timer;

        float exaggeration = iteration_counter < 250 ? 16.0f : 4.0f;

        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        { 
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it) 
            {
                #ifdef INDEX_TRACKER
                TsnePoint2D& pointR = embeddedPoints[indexTracker[it.row()]];
                TsnePoint2D& pointC = embeddedPoints[indexTracker[it.col()]];
                #else
                TsnePoint2D& pointR = embeddedPoints[it.row()];
                TsnePoint2D& pointC = embeddedPoints[it.col()];
                #endif

                glm::vec2 diff = pointR.position - pointC.position;
                float dist = diff.x * diff.x + diff.y * diff.y;

                pointC.derivative += exaggeration * static_cast<float>(it.value()) * (diff / (1.0f + dist));
            }
        }

        timer.endTimer("attractive");
    }

    void updateAttractive()
    {
        Timer timer;

        float exageration = iteration_counter < 250 ? 16.0f : 4.0f;

        for (int n = 0; n < embeddedPoints.size(); n++)
        {
            #ifdef INDEX_TRACKER
            TsnePoint2D& pointR = embeddedPoints[indexTracker[n]];
            #else
            TsnePoint2D& pointR = embeddedPoints[n];
            #endif

            glm::vec2 dim{ 0.0f };

            for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++)
            {
                unsigned int col = col_P[i];
                #ifdef INDEX_TRACKER
                TsnePoint2D& pointC = embeddedPoints[indexTracker[col]];
                #else
                TsnePoint2D& pointC = embeddedPoints[col];
                #endif

                glm::vec2 diff = pointC.position - pointR.position;
                float d_ij = diff.x * diff.x + diff.y * diff.y;
                float q_ij = 1.0f / (1.0f + d_ij);

                dim += exageration * static_cast<float>(val_P[i]) * q_ij * diff;
            }

            pointR.derivative += dim;
        }

        timer.endTimer("attractive");
    }

    void costFunction(const std::vector<TsnePoint2D>& points, const std::vector<int>& points_indices, Eigen::SparseMatrix<double>& Pmatrix)
    {
        double QijTotal = 0.0;

        const unsigned numThreads = std::thread::hardware_concurrency();
        const size_t chunkSize = (points.size() + numThreads - 1) / numThreads;

        std::vector<std::thread> threads;
        std::vector<double> localTotals(numThreads, 0.0);

        for (unsigned t = 0; t < numThreads; ++t)
        {
            size_t begin = t * chunkSize;
            size_t end = std::min(begin + chunkSize, points.size());

            threads.emplace_back
            (
                [&, t, begin, end]()
                {
                    double threadTotal = 0.0;

                    for (size_t i = begin; i < end; ++i)
                    {
                        for (size_t j = 0; j < points.size(); ++j)
                        {
                            if (i == j) continue;

                            #ifdef INDEX_TRACKER
                            const TsnePoint2D& point_i = points[points_indices[i]];
                            const TsnePoint2D& point_j = points[points_indices[j]];
                            #else
                            const TsnePoint2D& point_i = points[i];
                            const TsnePoint2D& point_j = points[j];
                            #endif


                            glm::vec2 diff = point_j.position - point_i.position;
                            float distance_squared = diff.x * diff.x + diff.y * diff.y;
                            localTotals[t] += static_cast<double>(1.0f / (1.0f + distance_squared));
                        }
                    }
                }
            );
        }

        for (std::thread& th : threads)
            th.join();

        QijTotal = std::accumulate(localTotals.begin(), localTotals.end(), 0.0);


        double totalCost = 0.0;
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                if (it.col() != it.row())
                {
                    #ifdef INDEX_TRACKER
                    const TsnePoint2D& point_col = points[points_indices[it.col()]];
                    const TsnePoint2D& point_row = points[points_indices[it.row()]];
                    #else
                    const TsnePoint2D& point_col = points[it.col()];
                    const TsnePoint2D& point_row = points[it.row()];
                    #endif

                    glm::vec2 diff = point_col.position - point_row.position;
                    float distance_squared = diff.x * diff.x + diff.y * diff.y;
                    double Qij = static_cast<double>(1.0f / (1.0f + distance_squared)) / QijTotal;

                    double Pij = it.value();

                    totalCost += Pij * std::log2(Pij / Qij);
                }
            }
        }

        std::cout << "totalCost: " << totalCost << std::endl;
    }

    void setupEfficientPmatrix()
    {
        //const double* values = Pmatrix.valuePtr();
        //const int* innerIndices = Pmatrix.innerIndexPtr();
        //const int* outerStarts = Pmatrix.outerIndexPtr();
        //int nnz = Pmatrix.nonZeros();
        //int rows = Pmatrix.rows();

        //val_P.assign(values, values + nnz);
        //col_P.assign(innerIndices, innerIndices + nnz);
        //row_P.assign(outerStarts, outerStarts + rows + 1);

        std::size_t N = embeddedPoints.size();

        row_P = std::vector<unsigned int>(N + 1, 0u);
        col_P = std::vector<unsigned int>(Pmatrix.nonZeros(), 0u);
        val_P = std::vector<double>(Pmatrix.nonZeros(), 0u);

        int PmatrixCounter = 0;

        std::vector<SparseEntryCOO2D> sparse_matrix_COO(Pmatrix.nonZeros());
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                sparse_matrix_COO[PmatrixCounter] = SparseEntryCOO2D(it.col(), it.row(), it.value());

                PmatrixCounter++;
            }
        }
        std::sort
        (
            sparse_matrix_COO.begin(),
            sparse_matrix_COO.end(),
            [](const SparseEntryCOO2D& a, const SparseEntryCOO2D& b) -> bool
            {
                if (a.row != b.row)
                    return a.row < b.row;
                else
                    return a.col < b.col;
            }
        );
        row_P[0] = 0;
        int entryCounter = 0;
        for (int r = 0; r < N; r++) // go over each row
        {
            int amount_in_row = 0;

            while (entryCounter < Pmatrix.nonZeros() && sparse_matrix_COO[entryCounter].row == r)
            {
                col_P[entryCounter] = sparse_matrix_COO[entryCounter].col;
                val_P[entryCounter] = sparse_matrix_COO[entryCounter].val;
                amount_in_row++;
                entryCounter++;
            }

            row_P[r + 1] = row_P[r] + amount_in_row;
        }
    }
};