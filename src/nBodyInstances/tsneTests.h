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
#include <thread>
#include <atomic>
#include <numeric>

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
#include "../nbodysolvers/cpu/nBodySolverFMM_MORTON.h"
#include "../nbodysolvers/cpu/nBodySolverPM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
#include "../ffthelper.h"
#include "../Timer.h"
#include "../particleMesh.h"
#include "tsne.h"

class TsneTest
{
public:
    std::filesystem::path root_folder = std::filesystem::current_path();

	TsneTest() {}

	void errorTimestepTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, double theta, double cell_size, unsigned int seed)
    {
        TSNE tsne
        (
            theta,
            theta,
            cell_size,
            dataset_type,
            data_size,
            perplexity_value,
            seed
        );

        // set graph size
        #ifndef INDEX_TRACKER
        std::vector<double> errorBH(iteration_amount, 0.0f);
        std::vector<double> errorBHMP(iteration_amount, 0.0f);
        std::vector<double> errorBHR(iteration_amount, 0.0f);
        std::vector<double> errorBHRMP(iteration_amount, 0.0f);
        std::vector<double> errorFMM(iteration_amount, 0.0f);
        std::vector<double> errorPM(iteration_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> errorFMM_MORTON(iteration_amount, 0.0f);
        std::vector<double> errorFMM_SYM_MORTON(iteration_amount, 0.0f);
        #endif
        #ifndef INDEX_TRACKER
        std::vector<int> timeBH(iteration_amount, 0);
        std::vector<int> timeBHMP(iteration_amount, 0);
        std::vector<int> timeBHR(iteration_amount, 0);
        std::vector<int> timeBHRMP(iteration_amount, 0);
        std::vector<int> timeFMM(iteration_amount, 0);
        std::vector<int> timePM(iteration_amount, 0);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<int> timeFMM_MORTON(iteration_amount, 0);
        std::vector<int> timeFMM_SYM_MORTON(iteration_amount, 0);
        #endif


        std::vector<TsnePoint2D> naiveSolution(data_size);
        std::vector<TsnePoint2D> fastSolution(data_size);
        std::vector<int> naiveSolutionIndexTracker(data_size);
        std::vector<int> fastSolutionIndexTracker(data_size);

        #ifdef INDEX_TRACKER
        std::string indexCheck = "index";
        #else
        std::string indexCheck = "noindex";
        #endif
        std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck;
        if (!std::filesystem::exists(pre_computed_path))
        {
            std::filesystem::create_directories(pre_computed_path);
        }
        
        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            std::cout << "find error at iteration: " << t << std::endl;

            tsne.updateMinMaxPos();

            // correct solution up to machine precision
            if (std::filesystem::exists(pre_computed_path / std::to_string(t)))
            {
                #ifndef INDEX_TRACKER
                std::vector<int> naiveSolutionIndexTracker;
                #endif

                loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(t));
            }
            else
            {
                tsne.resetDeriv();
                tsne.iteration_counter = t;
                tsne.nBodySelect = "naive";
                tsne.updateDerivative();
                naiveSolution = tsne.embeddedPoints;
                #ifdef INDEX_TRACKER
                naiveSolutionIndexTracker = tsne.indexTracker;
                #else
                naiveSolutionIndexTracker = std::vector<int>();
                #endif
                storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(t));
            }

            #ifndef INDEX_TRACKER
            // calculate the result of every approximation technique and find the error by comparing to naive
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BH";
            tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorBH[t] = getNMAE(naiveSolution, fastSolution);
            timeBH[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHMP";
            tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorBHMP[t] = getNMAE(naiveSolution, fastSolution);
            timeBHMP[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHR";
            tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorBHR[t] = getNMAE(naiveSolution, fastSolution);
            timeBHR[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "BHRMP";
            tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorBHRMP[t] = getNMAE(naiveSolution, fastSolution);
            timeBHRMP[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "FMM";
            tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorFMM[t] = getNMAE(naiveSolution, fastSolution);
            timeFMM[t] = t;

            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "PM";
            tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            //fastSolutionIndexTracker = tsne.indexTracker;
            errorPM[t] = getNMAE(naiveSolution, fastSolution);
            timePM[t] = t;
            #endif

            #ifdef INDEX_TRACKER
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
            #endif


            // update points with naive solution
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = "naive";
            tsne.embeddedPoints = naiveSolution;
            #ifdef INDEX_TRACKER
            tsne.indexTracker = naiveSolutionIndexTracker;
            #endif
            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        #ifndef INDEX_TRACKER
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
        #endif
        #ifdef INDEX_TRACKER
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM_MORTON, errorFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + fltToStr(theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM_SYM_MORTON, errorFMM_SYM_MORTON, methodPath);
        #endif
    }


    void calculationtimeThetaTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, std::vector<double> thetas, std::vector<double> cell_sizes, unsigned int seed)
    {
        int theta_diversity_amount = thetas.size();

        //std::vector<double> calculationtimeNaive(theta_diversity_amount, 0.0f);
        #ifndef INDEX_TRACKER
        std::vector<double> calculationtimeBH(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHR(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeFMM(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimePM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> calculationtimeFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif
    
        //std::vector<float> thetaNaive(theta_diversity_amount, 0.0f);
        #ifndef INDEX_TRACKER
        std::vector<double> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<double> thetaPM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif
    
    
        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosenTheta = thetas[t];
            double chosenCellSize = cell_sizes[t];
            //std::cout << "chosenTheta: " << chosenTheta << std::endl;

            TSNE tsne
            (
                chosenTheta,
                chosenTheta,
                chosenCellSize,
                dataset_type,
                data_size,
                perplexity_value,
                seed
            );

            #ifndef INDEX_TRACKER
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaPM[t] = chosenCellSize;
            #endif
            #ifdef INDEX_TRACKER
            thetaFMM_MORTON[t] = chosenTheta;
            thetaFMM_SYM_MORTON[t] = chosenTheta;
            #endif

            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);


            #ifdef INDEX_TRACKER
            std::string indexCheck = "index";
            #else
            std::string indexCheck = "noindex";
            #endif
            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck;
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }

    
            double timeBefore = 0.0;
            for(int j = 0; j < iteration_amount; j++)
            { 
                //std::vector<TsnePoint2D>& naiveSolution = naiveSolutions[j];
                //std::vector<int>& naiveSolutionIndexTracker = naiveSolutionIndexTrackers[j];

                std::cout << "find calc time at iteration: " << j << ", for theta: " << chosenTheta << std::endl;

                tsne.updateMinMaxPos();

                //correct solution up to machine precision
                if (std::filesystem::exists(pre_computed_path / std::to_string(j)))
                {
                    loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                }
                else
                //if (naiveSolution.size() == 0)
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.updateDerivative();
                    naiveSolution = tsne.embeddedPoints;
                    #ifdef INDEX_TRACKER
                    naiveSolutionIndexTracker = tsne.indexTracker;
                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    #endif
                }

                #ifndef INDEX_TRACKER
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
                #endif

                #ifdef INDEX_TRACKER
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
                #endif
    


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                #ifdef INDEX_TRACKER
                tsne.indexTracker = naiveSolutionIndexTracker;
                #endif
                tsne.updatePoints();
            }
    
            //calculationtimeNaive[t]          /= static_cast<double>(iteration_amount);
            #ifndef INDEX_TRACKER
            calculationtimeBH[t]             /= static_cast<double>(iteration_amount);
            calculationtimeBHMP[t]           /= static_cast<double>(iteration_amount);
            calculationtimeBHR[t]            /= static_cast<double>(iteration_amount);
            calculationtimeBHRMP[t]          /= static_cast<double>(iteration_amount);
            calculationtimeFMM[t]            /= static_cast<double>(iteration_amount);
            calculationtimePM[t]             /= static_cast<double>(iteration_amount);
            #endif
            #ifdef INDEX_TRACKER
            calculationtimeFMM_MORTON[t]     /= static_cast<double>(iteration_amount);
            calculationtimeFMM_SYM_MORTON[t] /= static_cast<double>(iteration_amount);
            #endif
        }
    
        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        #ifndef INDEX_TRACKER
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
        #endif
        #ifdef INDEX_TRACKER
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_MORTON, calculationtimeFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeTheta" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_SYM_MORTON, calculationtimeFMM_SYM_MORTON, methodPath);
        #endif
    }


    void errorThetaTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, std::vector<float> thetas, std::vector<double> cell_sizes, unsigned int seed)
    {       
        int theta_diversity_amount = thetas.size();
        #ifndef INDEX_TRACKER
        std::vector<double> errorBH(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHR(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> errorFMM(theta_diversity_amount, 0.0f);
        std::vector<double> errorPM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> errorFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> errorFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif

        #ifndef INDEX_TRACKER
        std::vector<double> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<double> thetaPM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif


        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosenTheta = thetas[t];
            double chosenCellSize = cell_sizes[t];

            TSNE tsne
            (
                chosenTheta,
                chosenTheta,
                chosenCellSize,
                dataset_type,
                data_size,
                perplexity_value,
                seed
            );


            #ifndef INDEX_TRACKER
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaPM[t] = chosenCellSize;
            #endif
            #ifdef INDEX_TRACKER
            thetaFMM_MORTON[t] = chosenTheta;
            thetaFMM_SYM_MORTON[t] = chosenTheta;
            #endif

            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);
            std::vector<TsnePoint2D> fastSolution(data_size);
            std::vector<int> fastSolutionIndexTracker(data_size);

            #ifdef INDEX_TRACKER
            std::string indexCheck = "index";
            #else
            std::string indexCheck = "noindex";
            #endif
            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck;
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }

            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find error at iteration: " << j << ", for theta: " << chosenTheta << std::endl;

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
                    #ifdef INDEX_TRACKER
                    naiveSolutionIndexTracker = tsne.indexTracker;
                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    #endif
                    //std::cout << "i have stored " << j << std::endl;
                }

                #ifndef INDEX_TRACKER
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BH";
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBH[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHMP";
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHMP[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHR";
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHR[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHRMP";
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHRMP[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM";
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "PM";
                tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorPM[t] += getNMAE(naiveSolution, fastSolution);
                #endif

                #ifdef INDEX_TRACKER
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_MORTON";
                tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_MORTON[t] += getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_SYM_MORTON";
                tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_SYM_MORTON[t] += getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
                #endif

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                #ifdef INDEX_TRACKER
                tsne.indexTracker = naiveSolutionIndexTracker;
                #endif
                tsne.updatePoints();
            }

            #ifndef INDEX_TRACKER
            errorBH[t] /= static_cast<double>(iteration_amount);
            errorBHMP[t] /= static_cast<double>(iteration_amount);
            errorBHR[t] /= static_cast<double>(iteration_amount);
            errorBHRMP[t] /= static_cast<double>(iteration_amount);
            errorFMM[t] /= static_cast<double>(iteration_amount);
            errorPM[t] /= static_cast<double>(iteration_amount);
            #endif
            #ifdef INDEX_TRACKER
            errorFMM_MORTON[t] /= static_cast<double>(iteration_amount);
            errorFMM_SYM_MORTON[t] /= static_cast<double>(iteration_amount);
            #endif
        }

        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));
        
        std::filesystem::path methodPath;
        #ifndef INDEX_TRACKER
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
        #endif
        #ifdef INDEX_TRACKER
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_MORTON, errorFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneErrorTheta" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile(thetaFMM_SYM_MORTON, errorFMM_SYM_MORTON, methodPath);
        #endif
    }


    void calculationtimeErrorTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, std::vector<double> thetas, std::vector<double> cell_sizes, unsigned int seed)
    {
        int theta_diversity_amount = thetas.size();
        #ifndef INDEX_TRACKER
        std::vector<double> calculationtimeBH(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHR(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeFMM(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimePM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> calculationtimeFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> calculationtimeFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif

        #ifndef INDEX_TRACKER
        std::vector<double> errorBH(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHR(theta_diversity_amount, 0.0f);
        std::vector<double> errorBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> errorFMM(theta_diversity_amount, 0.0f);
        std::vector<double> errorPM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> errorFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> errorFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif

        #ifndef INDEX_TRACKER
        std::vector<double> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<double> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM(theta_diversity_amount, 0.0f);
        std::vector<double> thetaPM(theta_diversity_amount, 0.0f);
        #endif
        #ifdef INDEX_TRACKER
        std::vector<double> thetaFMM_MORTON(theta_diversity_amount, 0.0f);
        std::vector<double> thetaFMM_SYM_MORTON(theta_diversity_amount, 0.0f);
        #endif


        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosenTheta = thetas[t];
            double chosenCellSize = cell_sizes[t];

            TSNE tsne
            (
                chosenTheta,
                chosenTheta,
                chosenCellSize,
                dataset_type,
                data_size,
                perplexity_value,
                seed
            );


            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);
            std::vector<TsnePoint2D> fastSolution(data_size);
            std::vector<int> fastSolutionIndexTracker(data_size);


            #ifdef INDEX_TRACKER
            std::string indexCheck = "index";
            #else
            std::string indexCheck = "noindex";
            #endif
            std::filesystem::path pre_computed_path = root_folder / std::string("precomputed_states") / dataset_type / std::to_string(data_size) / std::to_string(static_cast<unsigned int>(perplexity_value)) / std::to_string(seed) / indexCheck;
            if (!std::filesystem::exists(pre_computed_path))
            {
                std::filesystem::create_directories(pre_computed_path);
            }


            double timeBefore = 0.0f;
            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find calc time and error at iteration: " << j << ", for theta: " << chosenTheta << std::endl;

                tsne.updateMinMaxPos();

                // correct solution up to machine precision
                if (std::filesystem::exists(pre_computed_path / std::to_string(j)))
                {
                    #ifndef INDEX_TRACKER
                    std::vector<int> naiveSolutionIndexTracker;
                    #endif

                    loadPrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                }
                else
                {
                    tsne.resetDeriv();
                    tsne.iteration_counter = j;
                    tsne.nBodySelect = "naive";
                    tsne.updateDerivative();
                    naiveSolution = tsne.embeddedPoints;
                    #ifdef INDEX_TRACKER
                    naiveSolutionIndexTracker = tsne.indexTracker;
                    #else
                    naiveSolutionIndexTracker = std::vector<int>();
                    #endif
                    storePrecomputed(naiveSolution, naiveSolutionIndexTracker, pre_computed_path / std::to_string(j));
                    //std::cout << "i have stored " << j << std::endl;
                }
                
                #ifndef INDEX_TRACKER
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BH";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBH[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHMP[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHR";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHR[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "BHRMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorBHRMP[t] += getNMAE(naiveSolution, fastSolution);

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM[t] += getNMAE(naiveSolution, fastSolution);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "PM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["PM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimePM[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                //fastSolutionIndexTracker = tsne.indexTracker;
                errorPM[t] += getNMAE(naiveSolution, fastSolution);
                #endif

                #ifdef INDEX_TRACKER
                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_MORTON[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_MORTON[t] += getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);


                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "FMM_SYM_MORTON";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM_SYM_MORTON"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM_SYM_MORTON[t] += glfwGetTime() - timeBefore;
                fastSolution = tsne.embeddedPoints;
                fastSolutionIndexTracker = tsne.indexTracker;
                errorFMM_SYM_MORTON[t] += getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
                #endif

                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = "naive";
                tsne.embeddedPoints = naiveSolution;
                #ifdef INDEX_TRACKER
                tsne.indexTracker = naiveSolutionIndexTracker;
                #endif
                tsne.updatePoints();
            }

            #ifndef INDEX_TRACKER
            errorBH[t] /= iteration_amount;
            errorBHMP[t] /= iteration_amount;
            errorBHR[t] /= iteration_amount;
            errorBHRMP[t] /= iteration_amount;
            errorFMM[t] /= iteration_amount;
            errorPM[t] /= iteration_amount;
            #endif
            #ifdef INDEX_TRACKER
            errorFMM_MORTON[t] /= iteration_amount;
            errorFMM_SYM_MORTON[t] /= iteration_amount;
            #endif

            #ifndef INDEX_TRACKER
            calculationtimeBH[t] /= iteration_amount;
            calculationtimeBHMP[t] /= iteration_amount;
            calculationtimeBHR[t] /= iteration_amount;
            calculationtimeBHRMP[t] /= iteration_amount;
            calculationtimeFMM[t] /= iteration_amount;
            calculationtimePM[t] /= iteration_amount;
            #endif
            #ifdef INDEX_TRACKER
            calculationtimeFMM_MORTON[t] /= iteration_amount;
            calculationtimeFMM_SYM_MORTON[t] /= iteration_amount;
            #endif

            #ifndef INDEX_TRACKER
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
            thetaPM[t] = chosenCellSize;
            #endif
            #ifdef INDEX_TRACKER
            thetaFMM_MORTON[t] = chosenTheta;
            thetaFMM_SYM_MORTON[t] = chosenTheta;
            #endif
        }


        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        #ifndef INDEX_TRACKER
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
        #endif
        #ifdef INDEX_TRACKER
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("FMM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorFMM_MORTON, calculationtimeFMM_MORTON, thetaFMM_MORTON, methodPath);
        methodPath = projectFolder / ("tsneCalculationtimeError" + std::string("FMM_SYM_MORTON") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(errorFMM_SYM_MORTON, calculationtimeFMM_SYM_MORTON, thetaFMM_SYM_MORTON, methodPath);
        #endif
    }


    // set theta and PM stuff manually
    void costTimestepTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, double min_theta, double max_theta, double cell_size, unsigned int seed, std::string method)
    {
        TSNE tsne
        (
            min_theta,
            max_theta,
            cell_size,
            dataset_type,
            data_size,
            perplexity_value,
            seed
        );

        // set graph size
        std::vector<double> costs(iteration_amount, 0.0);

        std::vector<int> times(iteration_amount, 0);


        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            std::cout << "find error at iteration: " << t << std::endl;

            tsne.updateMinMaxPos();

            // calculate the result of every approximation technique and find the error by comparing to naive
            tsne.resetDeriv();
            tsne.iteration_counter = t;
            tsne.nBodySelect = method;
            tsne.nBodySolvers[method]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            if (t % 10 == 0)
            {
                #ifdef INDEX_TRACKER
                costs[t] = costFunction(tsne.embeddedPoints, tsne.indexTracker, tsne.Pmatrix);
                #else
                costs[t] = costFunction(tsne.embeddedPoints, tsne.Pmatrix);
                #endif

                if (t > 0)
                {
                    double prevCost = costs[t - 10];
                    double currCost = costs[t];

                    for (int i = 1; i < 10; ++i)
                    {
                        double alpha = static_cast<double>(i) / 10.0;
                        costs[t - 10 + i] = prevCost + alpha * (currCost - prevCost);
                    }
                }
            }
            times[t] = t;

            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder / ("costTimestep" + std::string(method) + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_minTheta" + fltToStr(min_theta) + "_maxTheta" + fltToStr(max_theta) + "_dataset" + dataset_type + ".csv");
        writeToFile(times, costs, methodPath);
    }

    void calculationtimeCostTSNE(std::string dataset_type, int data_size, float perplexity_value, int iteration_amount, std::vector<double> thetas, std::vector<double> cell_sizes, unsigned int seed, std::string method)
    {
        int theta_diversity_amount = thetas.size();
        std::vector<double> calculationtime(theta_diversity_amount, 0.0f);

        std::vector<double> finalCost(theta_diversity_amount, 0.0f);

        //std::vector<float> thetas(theta_diversity_amount, 0.0f);


        for (int t = 0; t < theta_diversity_amount; t++)
        {
            double chosenTheta = thetas[t];
            double chosenCellSize = cell_sizes[t];

            TSNE tsne
            (
                chosenTheta,
                2.0,
                chosenCellSize,
                dataset_type,
                data_size,
                perplexity_value,
                seed
            );


            double timeBefore = 0.0f;
            for (int j = 0; j < iteration_amount; j++)
            {
                std::cout << "find calc time and error at iteration: " << j << ", for theta: " << chosenTheta << std::endl;

                tsne.updateMinMaxPos();



                tsne.resetDeriv();
                tsne.iteration_counter = j;
                tsne.nBodySelect = method;
                timeBefore = glfwGetTime();
                tsne.nBodySolvers[method]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtime[t] += glfwGetTime() - timeBefore;


                tsne.updatePoints();
            }

            #ifdef INDEX_TRACKER
            finalCost[t] = costFunction(tsne.embeddedPoints, tsne.indexTracker, tsne.Pmatrix);
            #else
            finalCost[t] = costFunction(tsne.embeddedPoints, tsne.Pmatrix);
            #endif




            calculationtime[t] /= iteration_amount;
        }


        // write results to csv files
        std::filesystem::path projectFolder = root_folder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneCalculationtimeCost" + std::string(method) + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_dataset" + dataset_type + ".csv");
        writeToFile3(finalCost, calculationtime, thetas, methodPath);
    }

private:
    //mean relative error or maximum relative error
    #ifdef INDEX_TRACKER
    double getNMAE(const std::vector<TsnePoint2D>& points_naive, const std::vector<int>& points_naive_indices, const std::vector<TsnePoint2D>& points_approx, const std::vector<int>& points_approx_indices)
    #else
    double getNMAE(const std::vector<TsnePoint2D>& points_naive, const std::vector<TsnePoint2D>& points_approx)
    #endif
    {
        double MAE = 0.0;
        double norm = 0.0;
    
        for (int i = 0; i < points_naive.size(); i++)
        {
            #ifdef INDEX_TRACKER
            TsnePoint2D pointNaive = points_naive[points_naive_indices[i]];
            TsnePoint2D pointApprox = points_approx[points_approx_indices[i]];
            #else
            TsnePoint2D pointNaive = points_naive[i];
            TsnePoint2D pointApprox = points_approx[i];
            #endif

            //MAE += static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative));
            //norm += static_cast<double>(glm::length(pointNaive.derivative));

            MAE += static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative)) / static_cast<double>(glm::length(pointNaive.derivative));

            //MAE = std::max(MAE, static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative)) / static_cast<double>(glm::length(pointNaive.derivative)));
        }
    
        //double NMAE = MAE / norm;
        //return NMAE;

        double NMAE = MAE / points_naive.size();

        //double NMAE = MAE;
        return NMAE;
    }

    #ifdef INDEX_TRACKER
    double costFunction(const std::vector<TsnePoint2D>& points, const std::vector<int>& points_indices, Eigen::SparseMatrix<double>& Pmatrix)
    #else
    double costFunction(const std::vector<TsnePoint2D>& points, Eigen::SparseMatrix<double>& Pmatrix)
    #endif
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


                            glm::dvec2 diff = point_j.position - point_i.position;
                            double distance_squared = diff.x * diff.x + diff.y * diff.y;
                            localTotals[t] += static_cast<double>(1.0 / (1.0 + distance_squared));
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

                    glm::dvec2 diff = point_col.position - point_row.position;
                    double distance_squared = diff.x * diff.x + diff.y * diff.y;
                    double Qij = static_cast<double>(1.0 / (1.0 + distance_squared)) / QijTotal;

                    double Pij = it.value();

                    totalCost += Pij * std::log2(Pij / Qij);
                }
            }
        }

        //std::cout << "total cost: " << totalCost << std::endl;
        return totalCost;
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