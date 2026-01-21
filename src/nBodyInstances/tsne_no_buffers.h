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
#include "../nbodysolvers/cpu/nBodySolverTest.h"
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

    #ifdef INDEX_TRACKER
    std::vector<int> indexTracker;
    std::vector<int> indexTrackerPrev;
    #endif

    int follow = 1;
    glm::dvec2 minPos;
    glm::dvec2 maxPos;

    int nodeLevelToShow = 0;

    float forceSize = 1.0f;

    std::map<std::string, NBodySolver<TsnePoint2D>*> nBodySolvers;
    std::string nBodySelect = "naive"; // default selector

    double learnRate;
    double accelerationRate; // kinda irrelevant since this is based on iteration_counter

    std::vector<uint8_t> labels;
    Eigen::SparseMatrix<double> Pmatrix;
    std::vector<unsigned int> row_P;
    std::vector<unsigned int> col_P;
    std::vector<double> val_P;

    double desired_iteration_per_second; // limits the speed of tsne
    double time_since_last_iteration;

    int iteration_counter = 0; // keeps track of which iteration we are on

    Timer thousand_iteration_timer;
    bool reached_thousand_iterations = false;

    //double min_theta = 0.5;
    double min_theta = 0.75;
    //double max_theta = 2.0;
    double max_theta = 2.0;

    double cell_size = 1.0;

    TSNE_no_buffers()
    {
        int dataAmount = 70000;
        float perplexity = 30.0f;
        std::string dataSet = "MNIST_digits"; // "MNIST_digits", "MNIST_fashion", "mice_brain_cells", "CIFAR10"

        learnRate = static_cast<double>(dataAmount) / 15.0;

        desired_iteration_per_second = 0.0;

        time_since_last_iteration = 0.0;

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

        //errorCompare.resize(dataAmount);


        //srand(time(NULL));
        srand(296343u);
        double sizeParam = 2.0;
        for (int i = 0; i < dataAmount; i++)
        {
            double randX = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            double randY = 2.0 * ((double)rand() / RAND_MAX) - 1.0;


            while (pow(randX, 2.0) + pow(randY, 2.0) > 1.0)
            {
                randX = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
                randY = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            }


            glm::dvec2 pos = glm::dvec2(
                pow(sizeParam * randX, 1.0),
                pow(sizeParam * randY, 1.0)
            );

            int lab = labels[i];

            #ifdef INDEX_TRACKER
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            #else
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            #endif 

            #ifdef INDEX_TRACKER
            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
            #endif
        }


        setupEfficientPmatrix();


        updateMinMaxPos();


        double set_theta = 0.5;
        double set_cell_size = cell_size;
        int max_children_per_node = 16;
        nBodySolvers["naive"] = new NBodySolverNaive<TsnePoint2D>(&TSNEnaiveKernel);
        //nBodySolvers["test"] = new NBodySolverTest<TsnePoint2D>(&TSNEtestKernel);
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
        nBodySolvers["FMM_MORTON"] = new NBodySolverFMM_MORTON<TsnePoint2D>(&TSNEFMM_MORTONNNKernel, &TSNEFMM_MORTONPNKernel, &TSNEFMM_MORTONNPKernel, &TSNEFMM_MORTONPPKernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7, dataAmount), set_theta);
        nBodySolvers["FMM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        nBodySolvers["FMM_SYM_MORTON"] = new NBodySolverFMM_SYM_MORTON<TsnePoint2D>(&TSNE_FMM_SYM_MORTON_NN_Kernel, &TSNE_FMM_SYM_MORTON_PN_Kernel, &TSNE_FMM_SYM_MORTON_PP_Kernel, max_children_per_node, NBodySolverFMM_MORTON<TsnePoint2D>::getDepth(max_children_per_node * 0.7, dataAmount), set_theta);
        nBodySolvers["FMM_SYM_MORTON"]->updateTree(embeddedPoints, minPos, maxPos);
        #endif
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

    void resetTsne(std::string dataset_type, int data_size, float perplexity_value, double learn_rate, double theta, double set_cell_size, unsigned int seed)
    {
        learnRate = static_cast<double>(data_size) / 15.0;
        desired_iteration_per_second = 0.0;
        time_since_last_iteration = 0.0;

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



        //srand(time(NULL));
        srand(seed);
        double sizeParam = 2.0;
        for (int i = 0; i < data_size; i++)
        {
            float randX = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            float randY = 2.0 * ((double)rand() / RAND_MAX) - 1.0;


            while (pow(randX, 2.0) + pow(randY, 2.0) > 1.0)
            {
                randX = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
                randY = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            }


            glm::dvec2 pos = glm::dvec2(
                pow(sizeParam * randX, 1.0),
                pow(sizeParam * randY, 1.0)
            );

            int lab = labels[i];

            #ifdef INDEX_TRACKER
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab, i, 0.0);
            #else
            embeddedPoints[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            embeddedPointsPrevPrev[i] = TsnePoint2D(pos, glm::dvec2(0.0), lab);
            #endif      

            #ifdef INDEX_TRACKER
            indexTracker[i] = i;
            indexTrackerPrev[i] = i;
            #endif
        }


        setupEfficientPmatrix();


        updateMinMaxPos();


        setThetaForAll(theta, set_cell_size);
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
    }

    void timeStep()
    {
        if (iteration_counter == 0)
            thousand_iteration_timer.startTimer();

        if (iteration_counter == 1000 && !reached_thousand_iterations)
        {
            reached_thousand_iterations = true;
            desired_iteration_per_second = 0.0;
            thousand_iteration_timer.endTimer("A thousand iterations");
        }


        if (glfwGetTime() - time_since_last_iteration >= 1.0 / desired_iteration_per_second)
        {
            time_since_last_iteration = glfwGetTime();

            updateDerivative();

            updatePoints();

            nBodySolvers[nBodySelect]->updateTree(embeddedPoints, minPos, maxPos);
        }
    }

    void updatePoints()
    {
        embeddedPointsPrev.swap(embeddedPointsPrevPrev);
        embeddedPoints.swap(embeddedPointsPrev);

        thetaFunction();

        accelerationRate = 0.8;
        if (iteration_counter < 250)
        {
            accelerationRate = 0.5;
        }


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


        #ifdef INDEX_TRACKER
        indexTrackerPrev.swap(indexTracker);
        #endif


        updateMinMaxPos();


        iteration_counter++;
    }

    void thetaFunction()
    {
        //float falloff_strength = 5.0f;
        //float theta_result =
        //    (max_theta - min_theta) *
        //    std::max(1.0f - std::pow(static_cast<float>(iteration_counter) / 1000.0f, falloff_strength), 0.0f) +
        //    min_theta;

        double sharpness = 1.0;
        double extra_theta_ratio = 0.5 + 0.5 * std::cos
        (
            std::numbers::pi *
            std::min
            (
                std::pow(static_cast<double>(iteration_counter) / 999.0, sharpness),
                1.0
            )
        );
        double theta_diff = max_theta - min_theta;
        double theta_result = min_theta + extra_theta_ratio * theta_diff;

        std::cout << "set theta to: " << theta_result << std::endl;
        std::cout << "set cell size to: " << cell_size << std::endl;

        setThetaForAll(theta_result, cell_size);
    }

    void setMinMaxTheta(double set_min_theta, double set_max_theta)
    {
        min_theta = set_min_theta;
        max_theta = set_max_theta;
    }

    void setThetaForAll(double new_theta, double new_cell_size)
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
        minPos = glm::dvec2(std::numeric_limits<double>::max());
        maxPos = glm::dvec2(std::numeric_limits<double>::lowest());

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
            embeddedPoint.derivative = glm::dvec2(0.0);
    }

    void updateRepulsive()
    {
        double QijTotal = 0.0;

        #ifdef INDEX_TRACKER
        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints, indexTracker);
        #else
        nBodySolvers[nBodySelect]->solveNbody(QijTotal, embeddedPoints);
        #endif

        for (int i = 0; i < embeddedPoints.size(); i++)
        {
            embeddedPoints[i].derivative *= (4.0 / QijTotal);
        }
    }

    void updateAttractiveOUTDATED()
    {
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
    }

    void updateAttractive()
    {
        double exageration = iteration_counter < 250 ? 16.0 : 4.0; // the early exaggeration is 4.0f

        for (int n = 0; n < embeddedPoints.size(); n++)
        {
            #ifdef INDEX_TRACKER
            TsnePoint2D& pointR = embeddedPoints[indexTracker[n]];
            #else
            TsnePoint2D& pointR = embeddedPoints[n];
            #endif

            glm::dvec2 dim{ 0.0f };

            for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++)
            {
                unsigned int col = col_P[i];
                #ifdef INDEX_TRACKER
                TsnePoint2D& pointC = embeddedPoints[indexTracker[col]];
                #else
                TsnePoint2D& pointC = embeddedPoints[col];
                #endif

                glm::dvec2 diff = pointC.position - pointR.position;
                double d_ij = diff.x * diff.x + diff.y * diff.y;
                double q_ij = 1.0 / (1.0 + d_ij);

                dim += exageration * val_P[i] * q_ij * diff;
            }

            pointR.derivative += dim;
        }
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
        val_P = std::vector<double>(Pmatrix.nonZeros(), 0.0);

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
