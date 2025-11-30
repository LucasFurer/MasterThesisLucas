#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp> 
#include <glm/gtx/string_cast.hpp>
#include <utility>
#include <vector>
#include <Fastor/Fastor.h>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <algorithm>
#include <string>
#include <exception>
#include <thread>
#include <chrono>

#include "../../common.h"
#include "nBodySolver.h"
#include "../../trees/cpu/quadtreeFMM.h"
#include "../../trees/cpu/nodeFMM2D.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"

template <typename T>
struct MortonPoint 
{
    uint32_t morton;
    T point;
};

template <typename T>
struct lessthan
{
    inline bool operator()(const MortonPoint<T>& x, const MortonPoint<T>& y) const
    {
        return x.morton < y.morton;
    }
};

template <typename T>
struct rightshift
{
    inline uint32_t operator()(const MortonPoint<T>& x, const unsigned offset) const
    {
        return x.morton >> offset;
    }
};

template <typename T>
class NBodySolverFMM : public NBodySolver<T>
{
public:
    unsigned int treeDepth;
    std::vector<NodeFMM2D> nodes;
    std::vector<unsigned int> levelIndex;
    std::vector<unsigned int> levelSize;
    std::vector<unsigned int> levelGridWidth;

    std::function<void(float&, NodeFMM2D&, NodeFMM2D&)> kernelNN;
    std::function<void(float&, T&, NodeFMM2D&)> kernelPN;
    std::function<void(float&, NodeFMM2D&, T&)> kernelNP;
    std::function<void(float&, T&, T&)> kernelPP;

    int FMMcounter = 0;
    int BHMPcounter = 0;
    int BHRMPcounter = 0;

    int NNinter = 0;
    int NPinter = 0;
    int PNinter = 0;
    int PPinter = 0;

    NBodySolverFMM() {}

    NBodySolverFMM
    (
        std::function<void(float&, NodeFMM2D&, NodeFMM2D&)> initKernelNN,
        std::function<void(float&, T&, NodeFMM2D&)> initKernelPN,
        std::function<void(float&, NodeFMM2D&, T&)> initKernelNP,
        std::function<void(float&, T&, T&)> initKernelPP,
        int initMaxChildren,
        unsigned int initTreeDepth,
        float initTheta
    )
    {
        kernelNN = initKernelNN;
        kernelPN = initKernelPN;
        kernelNP = initKernelNP;
        kernelPP = initKernelPP;
        this->maxChildren = initMaxChildren;
        initNodesSize(initTreeDepth);
        this->theta = initTheta;
    }
    
    void solveNbody(float& total, std::vector<T>& points) override
    {
        FMMcounter = 0;
        BHMPcounter = 0;
        BHRMPcounter = 0;

        NNinter = 0;
        NPinter = 0;
        PNinter = 0;
        PPinter = 0;

        traverseFMM(total, points, nodes[0], nodes[0], this->theta);

        std::cout << "FMMcounter: " << FMMcounter << std::endl;
        std::cout << "BHMPcounter: " << BHMPcounter << std::endl;
        std::cout << "BHRMPcounter: " << BHRMPcounter << std::endl;

        std::cout << "NNinter: " << NNinter << std::endl;
        std::cout << "NPinter: " << NPinter << std::endl;
        std::cout << "PNinter: " << PNinter << std::endl;
        std::cout << "PPinter: " << PPinter << std::endl;
        
        //for (int i = 0; i < points.size(); i++)
        //{
        //    traverseBHMP(total, points, points[i], nodes[0], this->theta);
        //}

        //for (int i = 0; i < points.size(); i++)
        //{
        //    traverseBHRMP(total, points, nodes[0], points[i], this->theta);
        //}

        applyForces(points, nodes[0]);
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
    {
        NodeFMM2D emptyNode;
        std::fill(nodes.begin(), nodes.end(), emptyNode);
        
        glm::vec2 negMinPos = -minPos;
        float largestAxis = glm::compMax(maxPos + negMinPos);

        
        std::vector<MortonPoint<T>> pointsMortons(points.size());
        for (int i = 0; i < points.size(); i++)
            pointsMortons[i] = MortonPoint<T>(createMortonCode(points[i].position, negMinPos, largestAxis), points[i]);
        boost::sort::spreadsort::integer_sort(pointsMortons.begin(), pointsMortons.end(), rightshift<T>(), lessthan<T>());
        //struct { bool operator()(MortonPoint<T> a, MortonPoint<T> b) const { return a.morton < b.morton; } } customLess;
        //std::sort(pointsMortons.begin(), pointsMortons.end(), customLess);
        for (int i = 0; i < points.size(); i++)
            points[i] = pointsMortons[i].point;

        createLeafNodes(points, minPos, maxPos);
        
        bottomUpNodeConstruction();

        //std::cout << "----------------------" << std::endl;
        //for (int i = 0; i < nodes.size(); i++)
        //{
        //    std::cout << "--node " << i << std::endl;
        //    std::cout << nodes[i].toString() << std::endl;
        //}
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        
        getNodesBufferData(result, nodes[0], 0, nodeLevelToShow);

        return result;
    }
    
private:   
    void traverseFMM(float& total, std::vector<T>& points, NodeFMM2D& sinkNode, NodeFMM2D& sourceNode, float theta)
    {
        FMMcounter++;
        std::cout << "----FMMcounter: " << FMMcounter << std::endl;
        //std::cout << "peorming with: " << std::endl;
        //std::cout << "sink: " << sinkNode.toString() << std::endl;
        //std::cout << "source: " << sourceNode.toString() << std::endl;
        //std::cout << "------------------------" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));

        glm::vec2 diff = sinkNode.centreOfMass - sourceNode.centreOfMass;
        float dist = glm::length(diff);

     
        if ((sinkNode.BBlength + sourceNode.BBlength) / dist < theta)
        {

            //if (sinkNode.particleIndexAmount != 0 && sourceNode.particleIndexAmount != 0)
            //{
            //    kernelNN(total, sinkNode, sourceNode);
            //    NNinter++;
            //}
            for (int sinkNodeChildIndex = sinkNode.firstChildIndex; sinkNodeChildIndex < sinkNode.firstChildIndex + 4; sinkNodeChildIndex++)
            {
                for (int sourceNodeChildIndex = sourceNode.firstChildIndex; sourceNodeChildIndex < sourceNode.firstChildIndex + 4; sourceNodeChildIndex++)
                {
                    if (nodes[sinkNodeChildIndex].particleIndexAmount != 0 && nodes[sourceNodeChildIndex].particleIndexAmount != 0)
                        traverseFMM(total, points, nodes[sinkNodeChildIndex], nodes[sourceNodeChildIndex], theta);

                }
            }

        }
        else if (sinkNode.firstChildIndex == 0)
        {

            for (int sinkNodePointIndex = sinkNode.firstParticleIndex; sinkNodePointIndex < sinkNode.firstParticleIndex + sinkNode.particleIndexAmount; sinkNodePointIndex++)
            {
                if (sourceNode.particleIndexAmount != 0)
                    traverseBHMP(total, points, points[sinkNodePointIndex], sourceNode, theta);

            }
        }
        else if (sourceNode.firstChildIndex == 0)
        {
            for (int sourceNodePointIndex = sourceNode.firstParticleIndex; sourceNodePointIndex < sourceNode.firstParticleIndex + sourceNode.particleIndexAmount; sourceNodePointIndex++)
            {
                if (sinkNode.particleIndexAmount != 0)
                    traverseBHRMP(total, points, sinkNode, points[sourceNodePointIndex], theta);

            }
        }
        else
        {
            for (int sinkNodeChildIndex = sinkNode.firstChildIndex; sinkNodeChildIndex < sinkNode.firstChildIndex + 4; sinkNodeChildIndex++)
            {
                for (int sourceNodeChildIndex = sourceNode.firstChildIndex; sourceNodeChildIndex < sourceNode.firstChildIndex + 4; sourceNodeChildIndex++)
                {

                    //if (nodes[sinkNodeChildIndex].firstChildIndex == 0 || nodes[sourceNodeChildIndex].firstChildIndex)
                    //{
                    //    std::cout << 
                    //}
                    //if (nodes[sinkNodeChildIndex].particleIndexAmount != 0 && nodes[sourceNodeChildIndex].particleIndexAmount != 0)
                    if (nodes[sinkNodeChildIndex].particleIndexAmount != 0 && nodes[sourceNodeChildIndex].particleIndexAmount != 0)
                        traverseFMM(total, points, nodes[sinkNodeChildIndex], nodes[sourceNodeChildIndex], theta);

                }
            }
        }

    }

    void traverseBHMP(float& total, std::vector<T>& points, T& sinkPoint, NodeFMM2D& sourceNode, float theta)
    {
        BHMPcounter++;
        std::cout << "----BHMPcounter: " << BHMPcounter << std::endl;
        //std::cout << "peorming with: " << std::endl;
        //std::cout << "sink: " << glm::to_string(sinkPoint.position) << std::endl;
        //std::cout << "source: " << sourceNode.toString() << std::endl;
        //std::cout << "------------------------" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));

        glm::vec2 diff = sinkPoint.position - sourceNode.centreOfMass;

        //std::cout << "sourceNode.BBlength / glm::length(diff) < theta: " << (sourceNode.BBlength / glm::length(diff) < theta) << std::endl;
        //std::cout << "sourceNode.firstChildIndex == 0: " << (sourceNode.firstChildIndex == 0) << std::endl;

        if (sourceNode.BBlength / glm::length(diff) < theta)
        {

            kernelPN(total, sinkPoint, sourceNode);
            PNinter++;

        }
        else if (sourceNode.firstChildIndex == 0)
        {
            for (int sourceNodePointIndex = sourceNode.firstParticleIndex; sourceNodePointIndex < sourceNode.firstParticleIndex + sourceNode.particleIndexAmount; sourceNodePointIndex++)
            {
                if (!glm::all(glm::equal(points[sourceNodePointIndex].position, sinkPoint.position)))
                {

                    kernelPP(total, sinkPoint, points[sourceNodePointIndex]);
                    PPinter++;

                }
            }
        }
        else
        {
            for (int sourceNodeChildIndex = sourceNode.firstChildIndex; sourceNodeChildIndex < sourceNode.firstChildIndex + 4; sourceNodeChildIndex++)
            {
                if (nodes[sourceNodeChildIndex].particleIndexAmount != 0)
                    traverseBHMP(total, points, sinkPoint, nodes[sourceNodeChildIndex], theta);

            }
        }
    }
    
    void traverseBHRMP(float& total, std::vector<T>& points, NodeFMM2D& sinkNode, T& sourcePoint, float theta)
    {
        BHRMPcounter++;
        std::cout << "----BHRMPcounter: " << BHRMPcounter << std::endl;
        //std::cout << "peorming with: " << std::endl;
        //std::cout << "sink: " << sinkNode.toString() << std::endl;
        //std::cout << "source: " << glm::to_string(sourcePoint.position) << std::endl;
        //std::cout << "------------------------" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));

        glm::vec2 diff = sinkNode.centreOfMass - sourcePoint.position;

        if (sinkNode.BBlength / glm::length(diff) < theta)
        {

            kernelNP(total, sinkNode, sourcePoint);
            NPinter++;

        }
        else if (sinkNode.firstChildIndex == 0)
        {
            for (int sinkNodePointIndex = sinkNode.firstParticleIndex; sinkNodePointIndex < sinkNode.firstParticleIndex + sinkNode.particleIndexAmount; sinkNodePointIndex++)
            {
                if (!glm::all(glm::equal(points[sinkNodePointIndex].position, sourcePoint.position)))
                {

                    kernelPP(total, points[sinkNodePointIndex], sourcePoint);
                    PPinter++;
                    
                }
            }
        }
        else
        {
            for (int sinkNodeChildIndex = sinkNode.firstChildIndex; sinkNodeChildIndex < sinkNode.firstChildIndex + 4; sinkNodeChildIndex++)
            {

                if (nodes[sinkNodeChildIndex].particleIndexAmount != 0)
                    traverseBHRMP(total, points, nodes[sinkNodeChildIndex], sourcePoint, theta);

            }
        }

    }
    
    void initNodesSize(unsigned int initTreeDepth)
    {
        treeDepth = initTreeDepth;
        levelIndex.resize(treeDepth + 1);
        levelSize.resize(treeDepth + 1);
        levelGridWidth.resize(treeDepth + 1);
        levelIndex[0] = 0;
        int nodesSize = 0;
        for (int i = 0; i <= treeDepth; i++) // treeDepth = 0 means just the root
        {
            int currentLevelSize = 1;
            for (int j = 0; j < i; j++)
            {
                currentLevelSize *= 4;
            }
            levelSize[i] = currentLevelSize;
            nodesSize += currentLevelSize;

            int currentDepthStart = 0;
            for (int j = 0; j <= i; j++)
            {
                currentDepthStart += levelSize[j];
            }
            if (i + 1 < treeDepth + 1)
                levelIndex[i + 1] = currentDepthStart;


        }

        nodes.resize(nodesSize);

        for (int i = 0; i <= treeDepth; i++)
            levelGridWidth[i] = std::pow(2, i);
        //leafGridSize = std::pow(2, treeDepth);
    }

    void createLeafNodes(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos)
    {
        glm::vec2 negMinPos = -minPos;
        float largestAxis = glm::compMax(maxPos + negMinPos);

        float leafNodeSize = largestAxis / static_cast<float>(levelGridWidth[treeDepth]);

        for (int i = 0; i < points.size(); i++)
        {
            glm::vec2 gridPos = static_cast<float>(levelGridWidth[treeDepth]) * (points[i].position + negMinPos) / (largestAxis);

            glm::vec<2, uint32_t> gridCoord = glm::min(glm::max(glm::vec<2, uint32_t>(gridPos), glm::vec<2, uint32_t>(0u)), glm::vec<2, uint32_t>(levelGridWidth[treeDepth] - 1u));

            uint32_t leafLevelIndex = (uint32_t)levelIndex[treeDepth] + ((spread16(gridCoord.y) << 1) | spread16(gridCoord.x));

            if (nodes[leafLevelIndex].particleIndexAmount == 0u)
            {
                nodes[leafLevelIndex].BBcentre = minPos + leafNodeSize * glm::vec2(gridCoord) + 0.5f * glm::vec2(leafNodeSize);
                nodes[leafLevelIndex].BBlength = leafNodeSize;

                nodes[leafLevelIndex].firstParticleIndex = i;
            }

            nodes[leafLevelIndex].particleIndexAmount += 1u;
            nodes[leafLevelIndex].centreOfMass += points[i].position;
            nodes[leafLevelIndex].M0 += 1.0f;
        }

        for (int i = levelIndex[treeDepth]; i < levelIndex[treeDepth] + levelSize[treeDepth]; i++)
        {
            if (nodes[i].M0 != 0.0f)
                nodes[i].centreOfMass /= nodes[i].M0;
        }
    }

    void bottomUpNodeConstruction()
    {
        std::array<glm::vec2, 4> BBcentreOffset
        {
            glm::vec2(0.5f, 0.5f),
            glm::vec2(-0.5f, 0.5f),
            glm::vec2(0.5f, -0.5f),
            glm::vec2(-0.5f, -0.5f)
        };

        for (int l = treeDepth-1; l >= 0; l--)
        {
            for (int i = 0; i < levelSize[l]; i++)
            {
                int nodeIndex = levelIndex[l] + i;

                unsigned int potentialFirstChildIndex = levelIndex[l + 1] + i * 4;
                //nodes[nodeIndex].firstChildIndex = levelIndex[l + 1] + i * 4;
                for (int j = 3; j >= 0; j--) // loop over all children of node i in level l
                {
                    //unsigned int childIndex = nodes[nodeIndex].firstChildIndex + j;
                    unsigned int childIndex = potentialFirstChildIndex + j;
                    if (nodes[childIndex].particleIndexAmount != 0u)
                    {
                        nodes[nodeIndex].firstChildIndex = potentialFirstChildIndex;

                        nodes[nodeIndex].BBcentre = BBcentreOffset[j] * nodes[childIndex].BBlength + nodes[childIndex].BBcentre;
                        nodes[nodeIndex].BBlength = 2.0f * nodes[childIndex].BBlength;

                        nodes[nodeIndex].firstParticleIndex = nodes[childIndex].firstParticleIndex;
                        nodes[nodeIndex].particleIndexAmount += nodes[childIndex].particleIndexAmount;

                        nodes[nodeIndex].centreOfMass += nodes[childIndex].M0 * nodes[childIndex].centreOfMass;

                        nodes[nodeIndex].M0 += nodes[childIndex].M0;
                    }
                }

                if (nodes[nodeIndex].particleIndexAmount != 0)
                    nodes[nodeIndex].centreOfMass /= nodes[nodeIndex].M0;
            }
        }
    }

    void applyForces(std::vector<T>& points, NodeFMM2D& node)
    {
        if (node.firstChildIndex != 0)
        {
            //for (QuadTreeFMM* child : children)
            for (int nodeIndex = node.firstChildIndex; nodeIndex < node.firstChildIndex + 4; nodeIndex++)
            {
                //NodeFMM2D& child = nodes[nodeIndex];

                if (nodes[nodeIndex].particleIndexAmount != 0)
                {
                    // prework
                    glm::vec2 oldZ = nodes[nodeIndex].centreOfMass;
                    glm::vec2 newZ = node.centreOfMass;
                    Fastor::Tensor<float, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y };
                    Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                    Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                    Fastor::Tensor<float, 2> newC1 = node.C1 +
                        Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                        (1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                    Fastor::Tensor<float, 2, 2> newC2 = node.C2 +
                        Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3);

                    Fastor::Tensor<float, 2, 2, 2> newC3 = node.C3;

                    // add translated C^n to child C^n
                    nodes[nodeIndex].C1 += newC1;
                    nodes[nodeIndex].C2 += newC2;
                    nodes[nodeIndex].C3 += newC3;

                    // try to apply forces for the child node
                    applyForces(points, nodes[nodeIndex]);
                }
            }
        }
        else
        {
            for (int pointIndex = node.firstParticleIndex; pointIndex < node.firstParticleIndex + node.particleIndexAmount; pointIndex++)
            {
                //T& point = points[pointIndex];

                // prework
                glm::vec2 x = points[pointIndex].position;
                glm::vec2 Z0 = node.centreOfMass;
                Fastor::Tensor<float, 2> diff1 = { x.x - Z0.x, x.y - Z0.y };
                Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
                Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

                Fastor::Tensor<float, 2> newC1 = node.C1 +
                    Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node.C2) +
                    (1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node.C3));

                points[pointIndex].derivative += glm::vec2(newC1(0), newC1(1));
            }
        }
    }

    uint32_t createMortonCode(glm::vec2 position, glm::vec2 negMinPos, float largestAxis)
    {
        position = (position + negMinPos) / (largestAxis); // rescale such that range is between [0-1]

        position = glm::min(glm::max(position * 65536.0f, glm::vec2(0.0f)), glm::vec2(65535.0f)); // rescale such that range is between [0-65535] which is the max number for 16 bits

        return (spread16((uint32_t)position.y) << 1) | spread16((uint32_t)position.x);
    }

    inline uint32_t spread16(uint32_t x)
    {
        x = (x | (x << 8)) & 0x00FF00FFu;
        x = (x | (x << 4)) & 0x0F0F0F0Fu;
        x = (x | (x << 2)) & 0x33333333u;
        x = (x | (x << 1)) & 0x55555555u;
        return x;
    }

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, NodeFMM2D node, int level, int showLevel)
    {
        if ((level == showLevel || showLevel == -1) && node.particleIndexAmount != 0)
        {
            const int colorsSize = 7;
            std::array<glm::vec3, colorsSize> colors{
                glm::vec3(1.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 0.0f, 1.0f),
                glm::vec3(1.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 1.0f),
                glm::vec3(1.0f, 0.0f, 1.0f)
            };

            glm::vec3 color = colors[std::min(showLevel + 1, colorsSize - 1)];

            glm::vec2 lowestCorner = node.BBcentre - 0.5f * node.BBlength;
            glm::vec2 highestCorner = node.BBcentre + 0.5f * node.BBlength;

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            //float crossSize = 0.1f;
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(crossSize, 0.0f), glm::vec3(1.0f,0.0f,0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(crossSize, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(0.0f, crossSize), glm::vec3(1.0f, 0.0f, 0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(0.0f, crossSize), glm::vec3(1.0f, 0.0f, 0.0f)));
        }

        if (node.firstChildIndex != 0)
        {
            for (int i = node.firstChildIndex; i < node.firstChildIndex + 4; i++)
            {
                getNodesBufferData(nodesBufferData, nodes[i], level + 1, showLevel);
            }
        }
    }

    void getNodesLeafBufferData(std::vector<VertexPos2Col3>& nodesBufferData)
    {
        for (int i = levelIndex[treeDepth]; i < levelIndex[treeDepth] + levelSize[treeDepth]; i++)
        {
            NodeFMM2D node = nodes[i];

            glm::vec2 lowestCorner = node.BBcentre - 0.5f * node.BBlength;
            glm::vec2 highestCorner = node.BBcentre + 0.5f * node.BBlength;

            glm::vec3 color(0.0f, 1.0f, 0.0f);

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
            nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

            //float crossSize = 0.1f;
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(crossSize, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(crossSize, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)));

            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass - glm::vec2(0.0f, crossSize), glm::vec3(1.0f, 0.0f, 0.0f)));
            //nodesBufferData.push_back(VertexPos2Col3(node.centreOfMass + glm::vec2(0.0f, crossSize), glm::vec3(1.0f, 0.0f, 0.0f)));
        }
    }
};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



//void TSNEFMMNNKernelNaive(float& total, NodeFMM2D& sinkNode, NodeFMM2D& sourceNode)
//{
//    glm::vec2 diff = sinkNode.centreOfMass - sourceNode.centreOfMass;
//    float dist = glm::length(diff);
//
//    float forceDecay = (1.0f / (1.0f + (dist * dist)));
//    total += sinkNode.M0 * sourceNode.M0 * forceDecay;
//
//    sinkNode.tempAccAcc += sourceNode.M0 * forceDecay * forceDecay * diff;
//}
void TSNEFMMNNKernel(float& total, NodeFMM2D& sinkNode, NodeFMM2D& sourceNode)
{
    glm::vec2 R = sinkNode.centreOfMass - sourceNode.centreOfMass;
    float r = glm::length(R);
    float rS = 1.0f + (r*r);
    
    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += (sinkNode.M0 * sourceNode.M0) / rS;

    float MA0 = sinkNode.M0;
    float MB0 = sourceNode.M0;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;

    // calculate the C^m
    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    Fastor::Tensor<float, 2, 2> C2 =
    {
        {
            MB0 * (D1 + R.x * R.x * D2),
            MB0 * (R.x * R.y * D2)
        },
        {
            MB0 * (R.y * R.x * D2),
            MB0 * (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 =
    {
        {
            {
                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                MB0 * ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                MB0 * ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                MB0 * ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                MB0 * ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                MB0 * ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                MB0 * ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };


    sinkNode.C1 += C1;
    sinkNode.C1 += C2;
    sinkNode.C1 += C3;
}


//void TSNEFMMPNKernelNaive(float& total, TsnePoint2D& sinkPoint, QuadTreeFMM<TsnePoint2D>* sourceNode)
//{
//    glm::vec2 diff = sinkPoint.position - sourceNode->centreOfMass;
//    float dist = glm::length(diff);
//
//    float forceDecay = (1.0f / (1.0f + (dist * dist)));
//    total += sourceNode->totalMass * forceDecay;
//
//    sinkPoint.derivative += sourceNode->totalMass * forceDecay * forceDecay * diff;
//}
void TSNEFMMPNKernel(float& total, TsnePoint2D& sinkPoint, NodeFMM2D& sourceNode)
{
    glm::vec2 R = sinkPoint.position - sourceNode.centreOfMass;
    float r = glm::length(R);
    float rS = 1.0f + (r * r);

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += sourceNode.M0 / rS;

    float MB0 = sourceNode.M0;
    Fastor::Tensor<float, 2, 2> MB2 = sourceNode.M2;
    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;


    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);

    Fastor::Tensor<float, 2> C1 =
    {
        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
    };

    sinkPoint.derivative += glm::vec2(C1(0), C1(1));
}


//void TSNEFMMNPKernelNaive(float& total, QuadTreeFMM<TsnePoint2D>* sinkNode, TsnePoint2D& sourcePoint)
//{
//    glm::vec2 diff = sinkNode->centreOfMass - sourcePoint.position; // change this
//    float dist = glm::length(diff);
//
//    float forceDecay = (1.0f / (1.0f + (dist * dist)));
//    total += sinkNode->totalMass * forceDecay;
//
//    sinkNode->tempAccAcc += forceDecay * forceDecay * diff;
//}
void TSNEFMMNPKernel(float& total, NodeFMM2D& sinkNode, TsnePoint2D& sourcePoint)
{
    glm::vec2 R = sinkNode.centreOfMass - sourcePoint.position;
    float r = glm::length(R);
    float rS = 1.0f + (r * r);

    float D1 = 1.0f / (rS * rS);
    float D2 = -4.0f / (rS * rS * rS);
    float D3 = 24.0f / (rS * rS * rS * rS);
    total += sinkNode.M0 / rS;


    Fastor::Tensor<float, 2> C1 =
    {
        (R.x * D1),
        (R.y * D1)
    };

    Fastor::Tensor<float, 2, 2> C2 =
    {
        {
            (D1 + R.x * R.x * D2),
            (R.x * R.y * D2)
        },
        {
            (R.y * R.x * D2),
            (D1 + R.y * R.y * D2)
        }
    };

    Fastor::Tensor<float, 2, 2, 2> C3 =
    {
        {
            {
                ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
                ((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
            },
            {
                ((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
                ((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
            }
        },
        {
            {
                ((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
                ((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
            },
            {
                ((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
                ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
            }
        }
    };

    sinkNode.C1 += C1;
    sinkNode.C2 += C2;
    sinkNode.C3 += C3;
}


void TSNEFMMPPKernel(float& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff);

    float forceDecay = 1.0f / (1.0f + (dist * dist));
    total += forceDecay;

    sinkPoint.derivative += forceDecay * forceDecay * diff;
}






// gravity kernals ----------------------------------------------------------------------------------------------------------------------



//void GRAVITYFMMNodeNodeKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveNode->centreOfMass - activeNode->centreOfMass;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    passiveNode->tempAccAcc += -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//void GRAVITYFMMNodeNodeKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, QuadTreeFMM<Particle2D>* activeNode)
//{
//    // prework
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveNode->centreOfMass - activeNode->centreOfMass;
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    float D1 =  -1.0f / (rS * rS * rS);
//    float D2 =   3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MA0 = passiveNode->totalMass;
//    float MB0 = activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
//    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;
//
//    // calculate the C^m
//    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//    Fastor::Tensor<float, 2> C1 =
//    {
//        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//    };
//
//    Fastor::Tensor<float, 2, 2> C2 =
//    {
//        {
//            MB0 * (D1 + R.x * R.x * D2),
//            MB0 * (R.x * R.y * D2)
//        },
//        {
//            MB0 * (R.y * R.x * D2),
//            MB0 * (D1 + R.y * R.y * D2)
//        }
//    };
//
//    Fastor::Tensor<float, 2, 2, 2> C3 =
//    {
//        {
//            {
//                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
//                MB0 * ((R.y) * D2             + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.y) * D2             + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
//                MB0 * ((R.x) * D2             + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
//            }
//        },
//        {
//            {
//                MB0 * ((R.y) * D2             + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
//                MB0 * ((R.x) * D2             + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.x) * D2             + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
//                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
//            }
//        }
//    };
//
//
//    passiveNode->C1 += C1;
//    passiveNode->C2 += C2;
//    passiveNode->C3 += C3;
//}
//
//
//glm::vec2 GRAVITYFMMParticleNodeKernalNaive(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveParticle.position - activeNode->centreOfMass;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    return -activeNode->totalMass * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//glm::vec2 GRAVITYFMMParticleNodeKernal(float* accumulator, Particle2D passiveParticle, QuadTreeFMM<Particle2D>* activeNode)
//{
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveParticle.position - activeNode->centreOfMass;
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    float D1 = -1.0f / (rS * rS * rS);
//    float D2 = 3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MB0 = activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2 = activeNode->quadrupole;
//    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;
//
//
//    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//    Fastor::Tensor<float, 2> C1 =
//    {
//        MB0 * (R.x * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f * (MB2TildeSum1)*D2 + 0.5f * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//    };
//
//    return glm::vec2(C1(0), C1(1));
//}
//
//
//void GRAVITYFMMNodeParticleKernalNaive(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveNode->centreOfMass - activeParticle.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = (1.0f / (distance + softening));
//    passiveNode->tempAccAcc += -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
//void GRAVITYFMMNodeParticleKernal(float* accumulator, QuadTreeFMM<Particle2D>* passiveNode, Particle2D activeParticle)
//{
//    // prework
//    float softening = 0.1f;
//
//    glm::vec2 R = passiveNode->centreOfMass - activeParticle.position; // dhenen
//    //glm::vec2 R = activeNode->centreOfMass - passiveNode->centreOfMass; // gadget4
//    float r = glm::length(R);
//    float rS = r + softening;
//
//    //float D0 = log(rS);
//    float D1 = -1.0f / (rS * rS * rS);
//    float D2 = 3.0f / (rS * rS * rS * rS * rS);
//    float D3 = -15.0f / (rS * rS * rS * rS * rS * rS * rS);
//
//    float MA0 = passiveNode->totalMass;
//    float MB0 = 1.0f; //activeNode->totalMass;
//    Fastor::Tensor<float, 2, 2> MB2{}; // = activeNode->quadrupole;
//    Fastor::Tensor<float, 2, 2> MB2Tilde = (1.0f / MB0) * MB2;
//
//    // calculate the C^m
//    float MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//    float MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//    float MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//    float MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//    Fastor::Tensor<float, 2> C1 =
//    {
//        MB0 * (R.x * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//        MB0 * (R.y * (D1 + 0.5f*(MB2TildeSum1)*D2 + 0.5f*(MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//    };
//
//    Fastor::Tensor<float, 2, 2> C2 =
//    {
//        {
//            MB0 * (D1 + R.x * R.x * D2),
//            MB0 * (R.x * R.y * D2)
//        },
//        {
//            MB0 * (R.y * R.x * D2),
//            MB0 * (D1 + R.y * R.y * D2)
//        }
//    };
//
//    Fastor::Tensor<float, 2, 2, 2> C3 =
//    {
//        {
//            {
//                MB0 * ((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
//                MB0 * ((R.y) * D2             + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.y) * D2             + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
//                MB0 * ((R.x) * D2             + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
//            }
//        },
//        {
//            {
//                MB0 * ((R.y) * D2             + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
//                MB0 * ((R.x) * D2             + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
//            },
//            {
//                MB0 * ((R.x) * D2             + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
//                MB0 * ((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
//            }
//        }
//    };
//
//    passiveNode->C1 += C1;
//    passiveNode->C2 += C2;
//    passiveNode->C3 += C3;
//}
//
//
//glm::vec2 GRAVITYFMMParticleParticleKernal(float* accumulator, Particle2D passiveParticle, Particle2D activeParticle)
//{
//    float softening = 0.1f;
//    glm::vec2 diff = passiveParticle.position - activeParticle.position;
//    float distance = glm::length(diff);
//
//    float oneOverDistance = 1.0f / (distance + softening);
//    return -1.0f * oneOverDistance * oneOverDistance * oneOverDistance * diff;
//}
