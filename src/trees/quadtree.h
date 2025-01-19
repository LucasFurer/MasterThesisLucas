#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include "../particles/embeddedPoint.h"

struct LineSegment2D
{
	glm::vec2 pointB;
	glm::vec2 pointE;
	glm::vec3 colorB;
	glm::vec3 colorE;
	int depth;

	LineSegment2D(glm::vec2 initPointB, glm::vec2 initPointA, glm::vec3 initColorB, glm::vec3 initColorE, int initDepth)
	{
		pointB = initPointB;
		pointE = initPointA;
		colorB = initColorB;
		colorE = initColorE;
		depth = initDepth;
	}

	LineSegment2D()
	{

	}

	static float* LineSegmentToFloat(LineSegment2D* lineSegments, std::size_t lineSegmentsSize)
	{
		int lineSegmentAmount = lineSegmentsSize / sizeof(LineSegment2D);

		float* result = new float[10 * lineSegmentAmount];

		for (int i = 0; i < lineSegmentAmount; i++)
		{
			result[10 * i + 0] = lineSegments[i].pointB.x;
			result[10 * i + 1] = lineSegments[i].pointB.y;

			result[10 * i + 2] = lineSegments[i].colorB.r;
			result[10 * i + 3] = lineSegments[i].colorB.g;
			result[10 * i + 4] = lineSegments[i].colorB.b;

			result[10 * i + 5] = lineSegments[i].pointE.x;
			result[10 * i + 6] = lineSegments[i].pointE.y;

			result[10 * i + 7] = lineSegments[i].colorE.r;
			result[10 * i + 8] = lineSegments[i].colorE.g;
			result[10 * i + 9] = lineSegments[i].colorE.b;
		}

		return result;
	}
};

class QuadTree
{
public:
	int maxChildren;
	std::vector<EmbeddedPoint>* allParticles;

	float totalMass;
	glm::vec2 centreOfMass;

	glm::vec2 lowestCorner;
	glm::vec2 highestCorner;

	std::vector<int> occupants;

	std::vector<QuadTree*> children;

	QuadTree(int initMaxChildren, std::vector<EmbeddedPoint>* initAllParticles)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		glm::vec2 setLowestCorner(std::numeric_limits<float>::infinity());
		glm::vec2 setHighestCorner(-std::numeric_limits<float>::infinity());

		for (int i = 0; i < allParticles->size(); i++)
		{
			occupants.push_back(i);

			setLowestCorner = glm::min(setLowestCorner, (*allParticles)[i].position);
			setHighestCorner = glm::max(setHighestCorner, (*allParticles)[i].position);
		}

		float largestDifference = std::max(setHighestCorner.x - setLowestCorner.x, setHighestCorner.y - setLowestCorner.y);
		setHighestCorner.x = setLowestCorner.x + largestDifference + 0.0001f;
		setHighestCorner.y = setLowestCorner.y + largestDifference + 0.0001f;

		lowestCorner = setLowestCorner;
		highestCorner = setHighestCorner;

		std::pair<float, glm::vec2> childMassPosition = createTree();
	}

	QuadTree(int initMaxChildren, std::vector<EmbeddedPoint>* initAllParticles, std::vector<int> initOccupants, glm::vec2 initLowestCorner, glm::vec2 initHighestCorner)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;

		//std::cout << "new node created" << std::endl;
	}

	std::pair<float, glm::vec2> createTree()
	{
		//std::cout << "createTree begin" << std::endl;
		//std::cout << occupants.size() << std::endl;
		if (occupants.size() > maxChildren)
		{
			//std::cout << "coondition" << std::endl;
			std::vector<int> HH;
			std::vector<int> HL;
			std::vector<int> LH;
			std::vector<int> LL;

			float l = (highestCorner.x - lowestCorner.x) / 2.0f;
			float middleX = lowestCorner.x + l;
			float middleY = lowestCorner.y + l;

			for (int i = 0; i < occupants.size(); i++)
			{
				int index = occupants[i];

				if ((*allParticles)[index].position.x >= middleX && (*allParticles)[index].position.y >= middleY)
				{
					HH.push_back(occupants[i]);
				}
				else if ((*allParticles)[index].position.x >= middleX && (*allParticles)[index].position.y < middleY)
				{
					HL.push_back(occupants[i]);
				}
				else if ((*allParticles)[index].position.x < middleX && (*allParticles)[index].position.y >= middleY)
				{
					LH.push_back(occupants[i]);
				}
				else if ((*allParticles)[index].position.x < middleX && (*allParticles)[index].position.y < middleY)
				{
					LL.push_back(occupants[i]);
				}
				else
				{
					std::cout << "something is wrong in octtree" << std::endl;
				}

				//std::cout << "the " << i << "th value has been added" << std::endl;
			}

			if (HH.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, HH, glm::vec2(middleX, middleY),               glm::vec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, HL, glm::vec2(middleX, lowestCorner.y),        glm::vec2(highestCorner.x, middleY))); }
			if (LH.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, LH, glm::vec2(lowestCorner.x, middleY),        glm::vec2(middleX, highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, LL, glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(middleX, middleY))); }

			/*
			std::cout << "lowest corner is: " << glm::to_string(lowestCorner) << std::endl;
			std::cout << "highest corner is: " << glm::to_string(highestCorner) << std::endl;

			std::cout << "occupants size is: " << occupants.size() << std::endl;
			std::cout << "HHH size is: " << HHH.size() << std::endl;
			std::cout << "HHL size is: " << HHL.size() << std::endl;
			std::cout << "HLH size is: " << HLH.size() << std::endl;
			std::cout << "LHH size is: " << LHH.size() << std::endl;
			std::cout << "HLL size is: " << HLL.size() << std::endl;
			std::cout << "LLH size is: " << LLH.size() << std::endl;
			std::cout << "LHL size is: " << LHL.size() << std::endl;
			std::cout << "LLL size is: " << LLL.size() << std::endl;
			*/

			for (QuadTree* octTree : children)
			{
				std::pair<float, glm::vec2> childMassPosition = octTree->createTree();
				totalMass += childMassPosition.first;
				centreOfMass += childMassPosition.first * childMassPosition.second;
			}

			centreOfMass /= totalMass;

			return std::make_pair(totalMass, centreOfMass);
		}
		else
		{
			//std::cout << "leaf" << std::endl;
			for (int i = 0; i < occupants.size(); i++)
			{
				//totalMass += allParticles[occupants[i]].mass;
				totalMass += 1.0f;
				//centreOfMass += allParticles[occupants[i]].mass * allParticles[occupants[i]].position;
				centreOfMass += 1.0f * (*allParticles)[occupants[i]].position;
			}

			centreOfMass /= totalMass;

			return std::make_pair(totalMass, centreOfMass);
		}
	}

	void getLineSegments(std::vector<LineSegment2D>& lineSegments, int level, int showLevel)
	{
		if (level == showLevel)
		{
			glm::vec3 color;

			switch (showLevel) {
			case 0:
				color = glm::vec3(1.0f, 0.0f, 0.0f);
				break;
			case 1:
				color = glm::vec3(0.0f, 1.0f, 0.0f);
				break;
			case 2:
				color = glm::vec3(0.0f, 0.0f, 1.0f);
				break;
			case 3:
				color = glm::vec3(1.0f, 1.0f, 0.0f);
				break;
			case 4:
				color = glm::vec3(0.0f, 1.0f, 1.0f);
				break;
			case 5:
				color = glm::vec3(1.0f, 0.0f, 1.0f);
				break;
			default:
				color = glm::vec3(1.0f, 1.0f, 1.0f);
			}

			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(highestCorner.x, lowestCorner.y),  color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(lowestCorner.x, highestCorner.y),  color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(lowestCorner.x, lowestCorner.y),   color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, highestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(lowestCorner.x, highestCorner.y),  color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y),  glm::vec2(lowestCorner.x, highestCorner.y),  color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(highestCorner.x, lowestCorner.y),  color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y),  glm::vec2(highestCorner.x, lowestCorner.y),  color, color, level));
		}


		for (QuadTree* octTree : children)
		{
			octTree->getLineSegments(lineSegments, level + 1, showLevel);
		}
	}

	~QuadTree()
	{

	}

private:
};


