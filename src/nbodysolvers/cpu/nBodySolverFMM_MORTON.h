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
#include "../../trees/cpu/nodeFMM_MORTON_2D.h"
#include "../../particles/embeddedPoint.h"
#include "../../particles/tsnePoint2D.h"
#include "../../particles/Particle2D.h"
#include "../../particles/morton_point.h"


template <typename T>
class NBodySolverFMM_MORTON : public NBodySolver<T>
{
public:
    unsigned int treeDepth;
    std::vector<NodeFMM_MORTON_2D> nodes;
    std::vector<unsigned int> levelIndex;
    std::vector<unsigned int> levelSize;
    std::vector<unsigned int> levelGridWidth;

    std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> kernelNN;
    std::function<void(double&, T&, NodeFMM_MORTON_2D&)> kernelPN;
    std::function<void(double&, NodeFMM_MORTON_2D&, T&)> kernelNP;
    std::function<void(double&, T&, T&)> kernelPP;

    NBodySolverFMM_MORTON() {}

    NBodySolverFMM_MORTON
    (
        std::function<void(double&, NodeFMM_MORTON_2D&, NodeFMM_MORTON_2D&)> initKernelNN,
        std::function<void(double&, T&, NodeFMM_MORTON_2D&)> initKernelPN,
        std::function<void(double&, NodeFMM_MORTON_2D&, T&)> initKernelNP,
        std::function<void(double&, T&, T&)> initKernelPP,
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
    
    void solveNbody(double& total, std::vector<T>& points, std::vector<int>& indexTracker) override
    {
        traverseFMM(total, points, nodes[0], nodes[0], this->theta);

        applyForces(points, nodes[0]);
    }

    void updateTree(std::vector<T>& points, glm::vec2 minPos, glm::vec2 maxPos) override
    {
        NodeFMM_MORTON_2D emptyNode;
        std::fill(nodes.begin(), nodes.end(), emptyNode);
        
        glm::vec2 negMinPos = -minPos;
        float largestAxis = glm::compMax(maxPos + negMinPos);
        

        std::vector<MortonPoint<T>> pointsMortons(points.size());
        for (int i = 0; i < points.size(); i++)
            pointsMortons[i] = MortonPoint<T>(createMortonCode(points[i].position, negMinPos, largestAxis), points[i]);
        boost::sort::spreadsort::integer_sort(pointsMortons.begin(), pointsMortons.end(), rightshift<T>(), lessthan<T>());
        for (int i = 0; i < points.size(); i++)
            points[i] = pointsMortons[i].point;


        createLeafNodes(points, minPos, maxPos);
        
        bottomUpNodeConstruction();
    }

    std::vector<VertexPos2Col3> getNodesBufferData(int nodeLevelToShow) override
    {
        std::vector<VertexPos2Col3> result;
        
        getNodesBufferData(result, nodes[0], 0, nodeLevelToShow);

        return result;
    }

    static unsigned int getDepth(int max_children, int N)
    {
        unsigned int depth = 0;
        while (N > max_children)
        {
            N /= 4;
            depth++;
        }

        return depth;
    }
    
private:   
    void traverseFMM(double& total, std::vector<T>& points, NodeFMM_MORTON_2D& sinkNode, NodeFMM_MORTON_2D& sourceNode, float theta)
    {
        glm::vec2 diff = sinkNode.centreOfMass - sourceNode.centreOfMass;
        float dist = glm::length(diff);
     
        if ((sinkNode.BBlength + sourceNode.BBlength) / dist < theta)
        {
            if (sinkNode.particleIndexAmount != 0 && sourceNode.particleIndexAmount != 0)
            {

                kernelNN(total, sinkNode, sourceNode);

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

                    if (nodes[sinkNodeChildIndex].particleIndexAmount != 0 && nodes[sourceNodeChildIndex].particleIndexAmount != 0)
                        traverseFMM(total, points, nodes[sinkNodeChildIndex], nodes[sourceNodeChildIndex], theta);

                }
            }
        }

    }

    void traverseBHMP(double& total, std::vector<T>& points, T& sinkPoint, NodeFMM_MORTON_2D& sourceNode, float theta)
    {
        glm::vec2 diff = sinkPoint.position - sourceNode.centreOfMass;
        float dist = glm::length(diff);

        if (sourceNode.BBlength / dist < theta)
        {

            kernelPN(total, sinkPoint, sourceNode);

        }
        else if (sourceNode.firstChildIndex == 0)
        {
            for (int sourceNodePointIndex = sourceNode.firstParticleIndex; sourceNodePointIndex < sourceNode.firstParticleIndex + sourceNode.particleIndexAmount; sourceNodePointIndex++)
            {
                if (!glm::all(glm::equal(points[sourceNodePointIndex].position, sinkPoint.position)))
                {

                    kernelPP(total, sinkPoint, points[sourceNodePointIndex]);

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
    
    void traverseBHRMP(double& total, std::vector<T>& points, NodeFMM_MORTON_2D& sinkNode, T& sourcePoint, float theta)
    {
        glm::vec2 diff = sinkNode.centreOfMass - sourcePoint.position;
        float dist = glm::length(diff);

        if (sinkNode.BBlength / dist < theta)
        {

            kernelNP(total, sinkNode, sourcePoint);

        }
        else if (sinkNode.firstChildIndex == 0)
        {
            for (int sinkNodePointIndex = sinkNode.firstParticleIndex; sinkNodePointIndex < sinkNode.firstParticleIndex + sinkNode.particleIndexAmount; sinkNodePointIndex++)
            {
                if (!glm::all(glm::equal(points[sinkNodePointIndex].position, sourcePoint.position)))
                {

                    kernelPP(total, points[sinkNodePointIndex], sourcePoint);
                    
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

        for (int n = levelIndex[treeDepth]; n < levelIndex[treeDepth] + levelSize[treeDepth]; n++)
        {
            if (nodes[n].M0 != 0.0f)
            {
                nodes[n].centreOfMass /= nodes[n].M0;

                for (int pointIndex = nodes[n].firstParticleIndex; pointIndex < nodes[n].firstParticleIndex + nodes[n].particleIndexAmount; pointIndex++)
                {
                    glm::vec2 relativeCoord = points[pointIndex].position - nodes[n].centreOfMass;

                    Fastor::Tensor<float, 2, 2> outer_product;
                    outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                    outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                    outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                    outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                    nodes[n].M2 += outer_product;
                }
            }
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

                for (int j = 3; j >= 0; j--) // loop over all children of node i in level l
                {
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
                {
                    nodes[nodeIndex].centreOfMass /= nodes[nodeIndex].M0;

                    for (int nodeChildIndex = nodes[nodeIndex].firstChildIndex; nodeChildIndex < nodes[nodeIndex].firstChildIndex + 4; nodeChildIndex++)
                    {
                        if (nodes[nodeChildIndex].M0 != 0.0f)
                        {
                            glm::vec2 relativeCoord = nodes[nodeChildIndex].centreOfMass - nodes[nodeIndex].centreOfMass;

                            Fastor::Tensor<float, 2, 2> outer_product;
                            outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
                            outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
                            outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
                            outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
                            nodes[nodeIndex].M2 += outer_product;

                            nodes[nodeIndex].M2 += nodes[nodeChildIndex].M2;
                        }
                    }
                }
            }
        }
    }

    void applyForces(std::vector<T>& points, NodeFMM_MORTON_2D& node)
    {
        if (node.firstChildIndex != 0)
        {
            for (int nodeIndex = node.firstChildIndex; nodeIndex < node.firstChildIndex + 4; nodeIndex++)
            {
                if (nodes[nodeIndex].particleIndexAmount != 0)
                {
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

                    nodes[nodeIndex].C1 += newC1;
                    nodes[nodeIndex].C2 += newC2;
                    nodes[nodeIndex].C3 += newC3;

                    applyForces(points, nodes[nodeIndex]);
                }
            }
        }
        else
        {
            for (int pointIndex = node.firstParticleIndex; pointIndex < node.firstParticleIndex + node.particleIndexAmount; pointIndex++)
            {
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

    void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, NodeFMM_MORTON_2D node, int level, int showLevel)
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
};



// TSNE kernals ----------------------------------------------------------------------------------------------------------------------



void TSNEFMM_MORTONNNKernel(double& total, NodeFMM_MORTON_2D& sinkNode, NodeFMM_MORTON_2D& sourceNode)
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
    sinkNode.C2 += C2;
    sinkNode.C3 += C3;
}


void TSNEFMM_MORTONPNKernel(double& total, TsnePoint2D& sinkPoint, NodeFMM_MORTON_2D& sourceNode)
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


void TSNEFMM_MORTONNPKernel(double& total, NodeFMM_MORTON_2D& sinkNode, TsnePoint2D& sourcePoint)
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


void TSNEFMM_MORTONPPKernel(double& total, TsnePoint2D& sinkPoint, TsnePoint2D& sourcePoint)
{
    glm::vec2 diff = sinkPoint.position - sourcePoint.position;
    float dist = glm::length(diff); // no need for length, we can do squared euclidean distance instead

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
