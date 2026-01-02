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


class TSNE_no_buffers
{
public:
    std::vector<TsnePoint2D> embeddedPoints;
    std::vector<TsnePoint2D> embeddedPointsPrev;
    std::vector<TsnePoint2D> embeddedPointsPrevPrev;

    std::vector<int> indexTracker;
    std::vector<int> indexTrackerPrev;

    int follow = 1;
    glm::vec2 minPos;
    glm::vec2 maxPos;

    int nodeLevelToShow = 0;

    float forceSize = 1.0f;

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect = "naive"; // default selector

    float learnRate;
    float accelerationRate; // kinda irrelevant since this is based on iteration_counter

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;

    float desired_iteration_per_second; // limits the speed of tsne
    float time_since_last_iteration;

    int iteration_counter = 0; // keeps track of which iteration we are on

    Timer thousand_iteration_timer;
    bool reached_thousand_iterations = false;

    //float min_theta = 0.5f;
    float min_theta = 0.5f;
    //float max_theta = 2.0f;
    float max_theta = 0.5f;

    TSNE_no_buffers()
    {
        int dataAmount = 1000;
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

        indexTracker.resize(dataAmount);
        indexTrackerPrev.resize(dataAmount);

        //errorCompare.resize(dataAmount);
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

            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);

            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
        }
        #ifdef TIME_ALGORITHM 
        time_point_creation.endTimer("tsne point creation");
        #endif

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("tsne update minmax");
        #endif

        float set_theta = 0.48f;
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
        nBodySolvers["PM"] = new NBodySolverPM<TsnePoint2D>(Pmatrix, embeddedPoints, 4, 0.9, 5);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_MORTON"] = new NBodySolverFMM_MORTON<TsnePoint2D>(&TSNEFMM_MORTONNNKernel, &TSNEFMM_MORTONPNKernel, &TSNEFMM_MORTONNPKernel, &TSNEFMM_MORTONPPKernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7f, dataAmount), set_theta);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"] = new NBodySolverFMM_SYM_MORTON<TsnePoint2D>(&TSNE_FMM_SYM_MORTON_NN_Kernel, &TSNE_FMM_SYM_MORTON_PN_Kernel, &TSNE_FMM_SYM_MORTON_PP_Kernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7f, dataAmount), set_theta);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
    }

    ~TSNE_no_buffers()
    {
        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            delete nBodySolverPointer.second;
        }

        nBodySolvers.clear();
    }

    void cleanup()
    {

    }

    void resetTsne(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, float theta, unsigned int seed)
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

        indexTracker.resize(data_size);
        indexTrackerPrev.resize(data_size);

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

            embeddedPoints[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::vec2(0.0f), lab, i);

            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
        }
        #ifdef TIME_ALGORITHM 
        time_point_creation.endTimer("tsne point creation");
        #endif

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("tsne update minmax");
        #endif

        setThetaForAll(theta);
        nBodySolvers["BH"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHR"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["BHRMP"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM"]->updateTree(embeddedPoints, minPos, maxPos);
        delete nBodySolvers["PM"];
        nBodySolvers["PM"] = new NBodySolverPM<TsnePoint2D>(Pmatrix, embeddedPoints, 4, 0.9, 5);
        nBodySolvers["PM"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
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

            #ifdef TIME_ALGORITHM 
            Timer time_calculate_derivative;
            #endif
            Timer derivative_timer;
            updateDerivative();
            derivative_timer.endTimer("update derivative");
            #ifdef TIME_ALGORITHM 
            time_calculate_derivative.endTimer("derivative calculation time");
            #endif

            updatePoints();

            #ifdef TIME_ALGORITHM 
            Timer time_update_tree;
            #endif
            nBodySolvers[nBodySelect]->updateTree(embeddedPoints, minPos, maxPos);

            #ifdef TIME_ALGORITHM 
            time_update_tree.endTimer("update tree");
            #endif

            #ifdef TIME_ALGORITHM 
            std::cout << "done with iteration-----------------" << std::endl;
            #endif

            time_step_timer.endTimer("timeStep");
        }
    }

    void updatePoints()
    {
        embeddedPointsPrev.swap(embeddedPointsPrevPrev);
        embeddedPoints.swap(embeddedPointsPrev);

        //thetaFunction();

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
            int indexPrev = indexTracker[i];
            int indexPrevPrev = indexTrackerPrev[i];

            embeddedPoints[indexPrev] = embeddedPointsPrev[indexPrev];
            embeddedPoints[indexPrev].position = embeddedPointsPrev[indexPrev].position + learnRate * embeddedPointsPrev[indexPrev].derivative + accelerationRate * (embeddedPointsPrev[indexPrev].position - embeddedPointsPrevPrev[indexPrevPrev].position);
        }
        #ifdef TIME_ALGORITHM 
        time_update.endTimer("update positions");
        #endif

        indexTrackerPrev.swap(indexTracker);

        #ifdef TIME_ALGORITHM 
        Timer time_update_minmax;
        #endif
        updateMinMaxPos();
        #ifdef TIME_ALGORITHM 
        time_update_minmax.endTimer("update minmax");
        #endif

        iteration_counter++;
    }

    void thetaFunction()
    {
        float falloff_strength = 5.0f;
        float theta_result =
            (max_theta - min_theta) *
            std::max(1.0f - std::pow(static_cast<float>(iteration_counter) / 1000.0f, falloff_strength), 0.0f) +
            min_theta;

        //std::cout << "set theta to: " << theta_result << std::endl;

        setThetaForAll(theta_result);
    }

    void setMinMaxTheta(float set_min_theta, float set_max_theta)
    {
        min_theta = set_min_theta;
        max_theta = set_max_theta;
    }

    void setThetaForAll(float new_theta)
    {
        for (std::pair<const std::string, NBodySolver<TsnePoint2D>*> nBodySolverPointer : nBodySolvers)
        {
            nBodySolverPointer.second->theta = new_theta;
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
        if (nBodySelect == "PM")
        {
            NBodySolverPM<TsnePoint2D>* PMpointer = dynamic_cast<NBodySolverPM<TsnePoint2D>*>(nBodySolvers["PM"]);
            PMpointer->iteration_counter = iteration_counter;

            double QijTotal = 0.0f;
            nBodySolvers["PM"]->solveNbody(QijTotal, embeddedPoints, indexTracker);
        }
        else
        {
            updateIndexTracker();

            resetDeriv();

            updateRepulsive();

            updateAttractive();
        }
    }

    void updateIndexTracker()
    {
        for (int i = 0; i < embeddedPoints.size(); i++)
            indexTracker[embeddedPoints[i].ID] = i;
    }

    void resetDeriv()
    {
        for (TsnePoint2D& embeddedPoint : embeddedPoints)
            embeddedPoint.derivative = glm::vec2(0.0f);
    }

    void updateRepulsive()
    {
        double QijTotal = 0.0f;

        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints, indexTracker);

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedPoints[i].derivative *= (4.0f / QijTotal);
        }
    }

    void updateAttractive()
    {
        for (int k = 0; k < Pmatrix.outerSize(); ++k) // https://stackoverflow.com/questions/22421244/eigen-package-iterate-over-row-major-sparse-matrix
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Pmatrix, k); it; ++it)
            {
                int indexR = indexTracker[it.row()];
                int indexC = indexTracker[it.col()];

                glm::vec2 diff = embeddedPoints[indexR].position - embeddedPoints[indexC].position;
                float dist = glm::length(diff);

                float exageration = 1.0f;
                if (iteration_counter < 250)
                {
                    exageration = 4.0f;
                }

                embeddedPoints[indexC].derivative += exageration * 4.0f * (float)it.value() * (diff / (1.0f + (dist * dist)));
            }
        }
    }
};
