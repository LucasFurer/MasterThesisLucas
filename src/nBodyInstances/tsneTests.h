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
#include "../nbodysolvers/cpu/nBodySolverPM.h"
#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
#include "../ffthelper.h"
#include "../Timer.h"
#include "../particleMesh.h"
#include "tsne.h"

class TsneTest
{
public:
	TsneTest() {}

	void errorTimestepTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta)
    {
        TSNE tsne;
        tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, theta);

        // set graph size
        std::vector<float> errorBH(iteration_amount, 0.0f);
        std::vector<float> errorBHMP(iteration_amount, 0.0f);
        std::vector<float> errorBHR(iteration_amount, 0.0f);
        std::vector<float> errorBHRMP(iteration_amount, 0.0f);
        std::vector<float> errorFMM(iteration_amount, 0.0f);
        std::vector<float> errorFMMiter(iteration_amount, 0.0f);

        std::vector<int> timeBH(iteration_amount, 0);
        std::vector<int> timeBHMP(iteration_amount, 0);
        std::vector<int> timeBHR(iteration_amount, 0);
        std::vector<int> timeBHRMP(iteration_amount, 0);
        std::vector<int> timeFMM(iteration_amount, 0);
        std::vector<int> timeFMMiter(iteration_amount, 0);

        //std::vector<std::vector<TsnePoint2D>> preComputedStates;
        std::vector<TsnePoint2D> naiveSolution(data_size);
        std::vector<int> naiveSolutionIndexTracker(data_size);
        std::vector<TsnePoint2D> fastSolution(data_size);
        std::vector<int> fastSolutionIndexTracker(data_size);

        // find error at every time step
        for (int t = 0; t < iteration_amount; t++)
        {
            // correct solution up to machine precision - precomputed and loaded from disk
            tsne.nBodySelect = "naive";
            tsne.updateDerivative();
            naiveSolution = tsne.embeddedPoints;
            naiveSolutionIndexTracker = tsne.indexTracker;

            // calculate the result of every approximation technique and find the error by comparing to naive
            tsne.nBodySelect = "BH";
            tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBH[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBH[t] = t;

            tsne.nBodySelect = "BHMP";
            tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHMP[t] = t;

            tsne.nBodySelect = "BHR";
            tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHR[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHR[t] = t;

            tsne.nBodySelect = "BHRMP";
            tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorBHRMP[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeBHRMP[t] = t;

            tsne.nBodySelect = "FMM";
            tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
            tsne.updateDerivative();
            fastSolution = tsne.embeddedPoints;
            fastSolutionIndexTracker = tsne.indexTracker;
            errorFMM[t] = getNMAE(naiveSolution, naiveSolutionIndexTracker, fastSolution, fastSolutionIndexTracker);
            timeFMM[t] = t;



            tsne.embeddedPoints = naiveSolution;
            tsne.indexTracker = naiveSolutionIndexTracker;
            tsne.updatePoints();
        }

        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        projectFolder = projectFolder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));


        std::filesystem::path methodPath;
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BH") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + std::to_string(static_cast<int>(theta)) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBH, errorBH, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + std::to_string(static_cast<int>(theta)) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHMP, errorBHMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHR") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + std::to_string(static_cast<int>(theta)) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHR, errorBHR, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("BHRMP") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + std::to_string(static_cast<int>(theta)) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeBHRMP, errorBHRMP, methodPath);
        methodPath = projectFolder / ("tsneErrorTimestep" + std::string("FMM") + "_point" + std::to_string(data_size) + "_perp" + std::to_string(static_cast<int>(perplexity_value)) + "_theta" + std::to_string(static_cast<int>(theta)) + "_dataset" + dataset_type + ".csv");
        writeToFile(timeFMM, errorFMM, methodPath);

        
        tsne.cleanup();
    }



    void calculationtimeThetaTSNE(std::string dataset_type, int data_size, float perplexity_value, float learn_rate, int iteration_amount, float theta_start, int theta_diversity_amount, float theta_range)
    {
        std::vector<float> calculationtimeNaive(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBH(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHR(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> calculationtimeFMM(theta_diversity_amount, 0.0f);
    
        std::vector<float> thetaNaive(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBH(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHR(theta_diversity_amount, 0.0f);
        std::vector<float> thetaBHRMP(theta_diversity_amount, 0.0f);
        std::vector<float> thetaFMM(theta_diversity_amount, 0.0f);
    
    
        for (int t = 0; t < theta_diversity_amount; t++)
        {
            float chosenTheta = ((float)t / theta_diversity_amount) * theta_range + theta_start;

            TSNE tsne;
            tsne.resetTsne(dataset_type, data_size, perplexity_value, learn_rate, chosenTheta);

    
            thetaNaive[t] = chosenTheta;
            thetaBH[t] = chosenTheta;
            thetaBHMP[t] = chosenTheta;
            thetaBHR[t] = chosenTheta;
            thetaBHRMP[t] = chosenTheta;
            thetaFMM[t] = chosenTheta;
   

            //std::vector<std::vector<TsnePoint2D>> preComputedStates;
            std::vector<TsnePoint2D> naiveSolution(data_size);
            std::vector<int> naiveSolutionIndexTracker(data_size);

    
            float timeBefore = 0.0f;
            for(int j = 0; j < iteration_amount; j++)
            { 
                tsne.nBodySelect = "naive";
                timeBefore = glfwGetTime();
                tsne.updateDerivative();
                calculationtimeNaive[t] += glfwGetTime() - timeBefore;
                naiveSolution = tsne.embeddedPoints;
                naiveSolutionIndexTracker = tsne.indexTracker;
    
                tsne.nBodySelect = "BH";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BH"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBH[t] += glfwGetTime() - timeBefore;
    
                tsne.nBodySelect = "BHMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHMP[t] += glfwGetTime() - timeBefore;
    
                tsne.nBodySelect = "BHR";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHR"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHR[t] += glfwGetTime() - timeBefore;
    
                tsne.nBodySelect = "FMM";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["FMM"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeFMM[t] += glfwGetTime() - timeBefore;

                tsne.nBodySelect = "BHRMP";
                timeBefore = glfwGetTime();
                tsne.nBodySolvers["BHRMP"]->updateTree(tsne.embeddedPoints, tsne.minPos, tsne.maxPos);
                tsne.updateDerivative();
                calculationtimeBHRMP[t] += glfwGetTime() - timeBefore;
    

    
                tsne.embeddedPoints = naiveSolution;
                tsne.indexTracker = naiveSolutionIndexTracker;
                tsne.updatePoints();
            }
    
            calculationtimeNaive[t] /= iteration_amount;
            calculationtimeBH[t]    /= iteration_amount;
            calculationtimeBHMP[t]  /= iteration_amount;
            calculationtimeBHR[t]   /= iteration_amount;
            calculationtimeBHRMP[t] /= iteration_amount;
            calculationtimeFMM[t]   /= iteration_amount;

            tsne.cleanup();
        }
    
        // write results to csv files
        std::filesystem::path projectFolder;
        #ifdef _WIN32
        projectFolder = std::filesystem::current_path();
        #endif
        #ifdef linux
        projectFolder = std::filesystem::current_path().parent_path();
        #endif
        projectFolder = projectFolder / "graphCSV" / dataset_type / std::to_string(data_size) / ("perp" + std::to_string(static_cast<int>(perplexity_value)));

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
    }



private:
    void copyPoints(const std::vector<TsnePoint2D>& src, const std::vector<int>& index_pointer_src, std::vector<TsnePoint2D>& dst)
    {
        for (int i = 0; i < src.size(); i++)
        {
            dst[i] = src[index_pointer_src[i]];
        }
    }

    float getNMAE(const std::vector<TsnePoint2D>& pointsNaive, const std::vector<int>& pointsNaiveIndexTracker, const std::vector<TsnePoint2D>& pointsApproximate, const std::vector<int>& pointsApproximateIndexTracker)
    {
        float NMAE = 0.0f;
        float divide = 0.0f;
    
        for (int i = 0; i < pointsNaive.size(); i++)
        {
            TsnePoint2D pointNaive = pointsNaive[pointsNaiveIndexTracker[i]];
            TsnePoint2D pointApproximate = pointsApproximate[pointsApproximateIndexTracker[i]];

            NMAE += powf(glm::length(pointNaive.derivative - pointApproximate.derivative), 1.0f);
            divide += powf(glm::length(pointNaive.derivative), 1.0f);
        }
    
        float NMSE = NMAE / divide;
        return NMSE;
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
};