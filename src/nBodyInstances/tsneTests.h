#pragma once

#define GLM_ENABLE_EXPERIMENTAL

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
#include <fstream>
#include <cstdint>

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
#include "../nbodysolvers/cpu/nBodySolverFMM_MORTON.h"
#include "../nbodysolvers/cpu/nBodySolverPM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
#include "../ffthelper.h"
#include "../Timer.h"
#include "../particleMesh.h"
#include "tsne.h"
#include "tsne_no_buffers.h"

class TsneTest
{
public:
    std::filesystem::path root_folder = std::filesystem::current_path();

	TsneTest() {}

	void errorTimestepTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta, int PM_grid_width, double cell_size, unsigned int seed)
    {
        TSNE tsne;
        tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, theta, seed);
        NBodySolverPM<TsnePoint2D>* PM = dynamic_cast<NBodySolverPM<TsnePoint2D>*>(tsne.nBodySolvers["PM"]);
        PM->C_min_num_intervals = PM_grid_width;
        PM->C_intervals_per_integer = cell_size;

        // set graph size
        std::vector<float> errorBH(iteration_amount, 0.0f);
        std::vector<float> errorBHMP(iteration_amount, 0.0f);
        std::vector<float> errorBHR(iteration_amount, 0.0f);
        std::vector<float> errorBHRMP(iteration_amount, 0.0f);
        std::vector<float> errorFMM(iteration_amount, 0.0f);
        std::vector<float> errorPM(iteration_amount, 0.0f);
        std::vector<float> errorFMM_MORTON(iteration_amount, 0.0f);
        std::vector<float> errorFMM_SYM_MORTON(iteration_amount, 0.0f);

        std::vector<int> timeBH(iteration_amount, 0);
        std::vector<int> timeBHMP(iteration_amount, 0);
        std::vector<int> timeBHR(iteration_amount, 0);
        std::vector<int> timeBHRMP(iteration_amount, 0);
        std::vector<int> timeFMM(iteration_amount, 0);
        std::vector<int> timePM(iteration_amount, 0);
        std::vector<int> timeFMM_MORTON(iteration_amount, 0);
        std::vector<int> timeFMM_SYM_MORTON(iteration_amount, 0);


        std::vector<TsnePoint2D> naiveSolution(data_size);
        std::vector<int> naiveSolutionIndexTracker(data_size);
        std::vector<TsnePoint2D> fastSolution(data_size);
        std::vector<int> fastSolutionIndexTracker(data_size);


        std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed);
        if (!std::filesystem::exists(pre_computed_path))
        {
            std::filesystem::create_directories(pre_computed_path);
        }
        
        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            tsne.updateMinMaxPos();

            // correct solution up to machine precision
            if (std::filesystem::exists(pre_computed_path / std::to_string(t)))
            {
                loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(t));
            }
            else
            {
                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naiveSolution = tsne.embeddedPoints;
                naiveSolutionIndexTracker = tsne.indexTracker;

                storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(t));
            }


            // calculate the result of every approximation technique and find the error by comparing to naive
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BH";
            tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBH[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBH[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHMP";
            tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHMP[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHR";
            tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHR[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHR[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHRMP";
            tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHRMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHRMP[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "FMM";
            tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorFMM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeFMM[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "PM";
            tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorPM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timePM[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "FMM_MORTON";
            tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorFMM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeFMM_MORTON[t] = t;
            tsne.embeddedPoints = naiveSolution;
            tsne.indexTracker = naiveSolutionIndexTracker;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "FMM_SYM_MORTON";
            tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorFMM_SYM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeFMM_SYM_MORTON[t] = t;
            tsne.embeddedPoints = naiveSolution;
            tsne.indexTracker = naiveSolutionIndexTracker;


            // update points with naive solution
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.embeddedPoints = naiveSolution;
            tsne.indexTracker = naiveSolutionIndexTracker;
            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BH") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBH, errorBH, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHMP, errorBHMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHR") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHR, errorBHR, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHRMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHRMP, errorBHRMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM, errorFMM, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("PM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timePM, errorPM, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM_MORTON, errorFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM_SYM_MORTON, errorFMM_SYM_MORTON, methodPath);

        
        tsne.cleanup();
    }


    void calculationtimeThetaTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta_start, int theta_diversity_amount, float theta_range, int PM_grid_width, double cell_size, unsigned int seed)
    {
        std::vector<float> calculationtimeNaive(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBH(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHR(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimePM(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
    
        std::vector<float> thetaNaive(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaPM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
    
    
        for (int t = 0; t < theta_diversity_amount; t++)
        {
            float chosenTheta = ((float)t / (float)(theta_diversity_amount-1)) * theta_range + theta_start;
            //std::cout << "chosenTheta: " << chosenTheta << std::endl;

            TSNE tsne;
            tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, chosenTheta, seed);
            NBodySolverPM<TsnePoint2D>* PM = dynamic_cast<NBodySolverPM<TsnePoint2D>*>(tsne.nBodySolvers["PM"]);
            PM->C_min_num_intervals = PM_grid_width;
            PM->C_intervals_per_integer = cell_size;



            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaFMM_MORTON[t] = chosenTheta;



            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);

            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed);
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }

    
            float timeBefore = 0.0f;
            for(int j = 0; j < iteration_amount; j++)
            { 
                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                if (std::filesystem::exists(pre_computed_path / std::to_string(j)))
                {
                    loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                }
                else
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.updateDerivative();
                    naiveSolution = tsne.embeddedPoints;
                    naiveSolutionIndexTracker = tsne.indexTracker;

                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                }

                // delete naive for performance
                //tsne.resetDeriv();
                //tsne.iteration_counter = j;
                //tsne.nBodySelect = "naive";
                //timeBefore = glfwGetTime();
                //tsne.updateDerivative();
                //calculationtimeNaive[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BH";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
    
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
    
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHR";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHRMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "PM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimePM[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_MORTON[t] += glfwGetTime() - timeBefore;

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_SYM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_SYM_MORTON[t] += glfwGetTime() - timeBefore;
    


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                tsne.indexTracker = naiveSolutionIndexTracker;
                tsne.updatePoints();
            }
    
            calculationtimeNaive[t]          /= iteration_amount;
            calculationtimeBH[t]             /= iteration_amount;
            calculationtimeBHMP[t]           /= iteration_amount;
            calculationtimeBHR[t]            /= iteration_amount;
            calculationtimeBHRMP[t]          /= iteration_amount;
            calculationtimeFMM[t]            /= iteration_amount;
            calculationtimePM[t]             /= iteration_amount;
            calculationtimeFMM_MORTON[t]     /= iteration_amount;
            calculationtimeFMM_SYM_MORTON[t] /= iteration_amount;

            tsne.cleanup();
        }
    
        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("Naive") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaNaive, calculationtimeNaive, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("BH") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBH, calculationtimeBH, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("BHMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHMP, calculationtimeBHMP, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("BHR") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHR, calculationtimeBHR, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("BHRMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHRMP, calculationtimeBHRMP, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("FMM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM, calculationtimeFMM, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("PM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaPM, calculationtimePM, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_MORTON, calculationtimeFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_SYM_MORTON, calculationtimeFMM_SYM_MORTON, methodPath);
    }


    void errorThetaTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta_start, int theta_diversity_amount, float theta_range, float PM_grid_size_start, float PM_grid_size_range, unsigned int seed)
    {       
        std::vector<float> errorBH(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHR(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM(theta_diversity_amount, 0.0f);
        std::vector<float> errorPM(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM_SYM_MORTON(theta_diversity_amount, 0.0f);

        std::vector<float> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaPM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);


        for (int t = 0; t < theta_diversity_amount; t++)
        {
            float chosenTheta = ((float)t / (float)(theta_diversity_amount - 1)) * theta_range + theta_start;
            float chosen_grid_size = ((float)t / (float)(theta_diversity_amount - 1)) * PM_grid_size_range + PM_grid_size_start;

            TSNE_no_buffers tsne;
            tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, chosenTheta, seed);
            NBodySolverPM<TsnePoint2D>* PM = dynamic_cast<NBodySolverPM<TsnePoint2D>*>(tsne.nBodySolvers["PM"]);
            PM->C_min_num_intervals = 1;
            PM->C_intervals_per_integer = chosen_grid_size;

            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaPM[t] = chosenTheta;
            thetaFMM_MORTON[t] = chosenTheta;
            thetaFMM_SYM_MORTON[t] = chosenTheta;

            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);
            std::vector<TsnePoint2D> fastSolution(data_size);
            std::vector<int> fastSolutionIndexTracker(data_size);

            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed);
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }

            for (int j = 0; j < iteration_amount; j++)
            {
                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                if (std::filesystem::exists(pre_computed_path / std::to_string(j)))
                {
                    //std::cout << "im about to load " << j << std::endl;
                    loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    //std::cout << "i have loaded " << j << std::endl;
                }
                else
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.updateDerivative();
                    naiveSolution = tsne.embeddedPoints;
                    naiveSolutionIndexTracker = tsne.indexTracker;

                    //std::cout << "im about to store " << j << std::endl;
                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    //std::cout << "i have stored " << j << std::endl;
                }


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BH";
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBH[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHMP";
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHR";
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHR[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHRMP";
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHRMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM";
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "PM";
                tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorPM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_MORTON";
                tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_SYM_MORTON";
                tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_SYM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                tsne.indexTracker = naiveSolutionIndexTracker;
                tsne.updatePoints();
            }

            errorBH[t] /= iteration_amount;
            errorBHMP[t] /= iteration_amount;
            errorBHR[t] /= iteration_amount;
            errorBHRMP[t] /= iteration_amount;
            errorFMM[t] /= iteration_amount;
            errorPM[t] /= iteration_amount;
            errorFMM_MORTON[t] /= iteration_amount;
            errorFMM_SYM_MORTON[t] /= iteration_amount;

            tsne.cleanup();
        }

        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));
        
        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("BH") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBH, errorBH, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("BHMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHMP, errorBHMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("BHR") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHR, errorBHR, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("BHRMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaBHRMP, errorBHRMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("FMM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM, errorFMM, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("PM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaPM, errorPM, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_MORTON, errorFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_SYM_MORTON, errorFMM_SYM_MORTON, methodPath);
    }


    void calculationtimeErrorTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta_start, int theta_diversity_amount, float theta_range, float PM_grid_size_start, float PM_grid_size_range, unsigned int seed)
    {
        std::vector<float> calculationtimeBH(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHR(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimePM(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM_SYM_MORTON(theta_diversity_amount, 0.0f);

        
        std::vector<float> errorBH(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHR(theta_diversity_amount, 0.0f);
        std::vector<float> errorBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM(theta_diversity_amount, 0.0f);
        std::vector<float> errorPM(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> errorFMM_SYM_MORTON(theta_diversity_amount, 0.0f);

        std::vector<float> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaPM(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);


        for (int t = 0; t < theta_diversity_amount; t++)
        {
            float chosenTheta = ((float)t / (float)(theta_diversity_amount - 1)) * theta_range + theta_start;
            float chosen_grid_size = ((float)t / (float)(theta_diversity_amount - 1)) * PM_grid_size_range + PM_grid_size_start;


            TSNE_no_buffers tsne;
            tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, chosenTheta, seed);
            NBodySolverPM<TsnePoint2D>* PM = dynamic_cast<NBodySolverPM<TsnePoint2D>*>(tsne.nBodySolvers["PM"]);
            PM->C_min_num_intervals = 1;
            PM->C_intervals_per_integer = chosen_grid_size;


            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);
            std::vector<TsnePoint2D> fastSolution(data_size);
            std::vector<int> fastSolutionIndexTracker(data_size);


            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed);
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }


            float timeBefore = 0.0f;
            for (int j = 0; j < iteration_amount; j++)
            {
                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                if (std::filesystem::exists(pre_computed_path / std::to_string(j)))
                {
                    //std::cout << "im about to load " << j << std::endl;
                    loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    //std::cout << "i have loaded " << j << std::endl;
                }
                else
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.updateDerivative();
                    naiveSolution = tsne.embeddedPoints;
                    naiveSolutionIndexTracker = tsne.indexTracker;

                    //std::cout << "im about to store " << j << std::endl;
                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    //std::cout << "i have stored " << j << std::endl;
                }
                

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BH";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBH[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHR";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHR[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHRMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorBHRMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "PM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimePM[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorPM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_MORTON[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_SYM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_SYM_MORTON[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_SYM_MORTON[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                tsne.indexTracker = naiveSolutionIndexTracker;
                tsne.updatePoints();
            }

            errorBH[t] /= iteration_amount;
            errorBHMP[t] /= iteration_amount;
            errorBHR[t] /= iteration_amount;
            errorBHRMP[t] /= iteration_amount;
            errorFMM[t] /= iteration_amount;
            errorPM[t] /= iteration_amount;
            errorFMM_MORTON[t] /= iteration_amount;
            errorFMM_SYM_MORTON[t] /= iteration_amount;

            calculationtimeBH[t] /= iteration_amount;
            calculationtimeBHMP[t] /= iteration_amount;
            calculationtimeBHR[t] /= iteration_amount;
            calculationtimeBHRMP[t] /= iteration_amount;
            calculationtimeFMM[t] /= iteration_amount;
            calculationtimePM[t] /= iteration_amount;
            calculationtimeFMM_MORTON[t] /= iteration_amount;
            calculationtimeFMM_SYM_MORTON[t] /= iteration_amount;

            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaPM[t] = chosen_grid_size;
            thetaFMM_MORTON[t] = chosenTheta;
            thetaFMM_SYM_MORTON[t] = chosenTheta;


            tsne.cleanup();
        }


        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("BH") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorBH, calculationtimeBH, thetaBH, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("BHMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorBHMP, calculationtimeBHMP, thetaBHMP, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("BHR") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorBHR, calculationtimeBHR, thetaBHR, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("BHRMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorBHRMP, calculationtimeBHRMP, thetaBHRMP, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("FMM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorFMM, calculationtimeFMM, thetaFMM, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("PM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorPM, calculationtimePM, thetaPM, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorFMM_MORTON, calculationtimeFMM_MORTON, thetaFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorFMM_SYM_MORTON, calculationtimeFMM_SYM_MORTON, thetaFMM_SYM_MORTON, methodPath);
    }

private:
    void copyPoints(const std::vector<TsnePoint2D>& src, const std::vector<int>& index_pointer_src, std::vector<TsnePoint2D>& dst)
    {
        for (int i = 0; i < src.size(); i++)
        {
            dst[i] = src[index_pointer_src[i]];
        }
    }

    float getNMAE(const std::vector<TsnePoint2D>& points_naive, const std::vector<int>& points_naive_indices, const std::vector<TsnePoint2D>& points_approx, const std::vector<int>& points_approx_indices)
    {
        float MAE = 0.0f;
        float norm = 0.0f;
    
        for (int i = 0; i < points_naive.size(); i++)
        {
            TsnePoint2D pointNaive = points_naive[points_naive_indices[i]];
            TsnePoint2D pointApprox = points_approx[points_approx_indices[i]];

            MAE += glm::length(pointNaive.derivative - pointApprox.derivative);
            norm += glm::length(pointNaive.derivative);
        }
    
        float NMAE = MAE / norm;
        return NMAE;
    }
    
    template <typename T, typename I>
    void writeToFile(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);
    
        if (!file.is_open()) 
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }
    
        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "\n";
    
        if (file.fail()) 
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else 
            std::cout << "File written successfully: " << filepath << std::endl;
    
        file.close();
    }
    
    template <typename T, typename I, typename J>
    void writeToFile3(const std::vector<T>& Xaxis, const std::vector<I>& Yaxis, const std::vector<J>& Zaxis, std::filesystem::path filepath)
    {
        std::ofstream file(filepath, std::ios::out);
    
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }
    
        for (size_t i = 0; i < Xaxis.size(); ++i)
            file << Xaxis[i] << "," << Yaxis[i] << "," << Zaxis[i] << "\n";
    
        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;
    
        file.close();
    }
    
    template <typename... Vectors>
    void writeToFileN(const std::filesystem::path& filepath, const Vectors&... vectors)
    {
        std::ofstream file(filepath, std::ios::out);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return;
        }
    
        size_t size = std::min({ vectors.size()... }); // take smallest size of vectors
    
        for (size_t i = 0; i < size; ++i)
        {
            std::ostringstream oss;
            ((oss << vectors[i] << ","), ...);
            std::string line = oss.str();
            file << line.substr(0, line.size() - 1) << "\n";
    
            //((file << vectors[i] << ((&vectors == &std::get<sizeof...(vectors) - 1>(std::tie(vectors...))) ? "\n" : ",")), ...);
        }
    
        if (file.fail())
            std::cerr << "Error: Failed while writing to file: " << filepath << std::endl;
        else
            std::cout << "File written successfully: " << filepath << std::endl;
    
        file.close();
    }
    
    std::string fltToStr(float f) 
    {
        std::ostringstream oss;
        oss << std::fixed << f;
        std::string str = oss.str();
    
        // remove zero's
        str.erase(str.find_last_not_of('0') + 1);
    
        // If it ends with a . remove it too
        if (!str.empty() && str.back() == '.') 
        {
            str.pop_back();
        }
    
        return str;
    }

    void loadPrecomputed(std::vector<TsnePoint2D>& P, std::vector<int>& I, std::filesystem::path path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Failed to open file for reading");
        }
        else
        {
            in.read(reinterpret_cast<char*>(P.data()), P.size() * sizeof(TsnePoint2D));
            in.read(reinterpret_cast<char*>(I.data()), I.size() * sizeof(int));
        }
    }

    void storePrecomputed(const std::vector<TsnePoint2D>& P, const std::vector<int>& I, std::filesystem::path path)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            throw std::runtime_error("Failed to open file for writing");
        }
        else
        {
            out.write(reinterpret_cast<const char*>(P.data()), P.size() * sizeof(TsnePoint2D));
            out.write(reinterpret_cast<const char*>(I.data()), I.size() * sizeof(int));
        }
    }
};