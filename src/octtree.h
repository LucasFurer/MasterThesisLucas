#ifndef OCTTREE_H
#define OCTTREE_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include "particle.h"

struct LineSegment
{
	glm::vec3 pointB;
	glm::vec3 pointE;
	glm::vec3 colorB;
	glm::vec3 colorE;
	int depth;

	LineSegment(glm::vec3 initPointB, glm::vec3 initPointA, glm::vec3 initColorB, glm::vec3 initColorE, int initDepth)
	{
		pointB = initPointB;
		pointE = initPointA;
		colorB = initColorB;
		colorE = initColorE;
		depth = initDepth;
	}

	LineSegment()
	{

	}

	static float* LineSegmentToFloat(LineSegment* lineSegments, std::size_t lineSegmentsSize)
	{
		int lineSegmentAmount = lineSegmentsSize / sizeof(LineSegment);

		float* result = new float[12 * lineSegmentAmount];

		for (int i = 0; i < lineSegmentAmount; i++)
		{
			result[12 * i + 0 ] = lineSegments[i].pointB.x;
			result[12 * i + 1 ] = lineSegments[i].pointB.y;
			result[12 * i + 2 ] = lineSegments[i].pointB.z;

			result[12 * i + 3 ] = lineSegments[i].colorB.r;
			result[12 * i + 4 ] = lineSegments[i].colorB.g;
			result[12 * i + 5 ] = lineSegments[i].colorB.b;

			result[12 * i + 6 ] = lineSegments[i].pointE.x;
			result[12 * i + 7 ] = lineSegments[i].pointE.y;
			result[12 * i + 8 ] = lineSegments[i].pointE.z;

			result[12 * i + 9 ] = lineSegments[i].colorE.r;
			result[12 * i + 10] = lineSegments[i].colorE.g;
			result[12 * i + 11] = lineSegments[i].colorE.b;
		}

		return result;
	}
};

class OctTree
{
public:
	static int maxChildren;
	static Particle* allParticles;
	static std::size_t allParticlesSize;

	float totalMass;
	glm::vec3 centreOfMass;

	glm::vec3 lowestCorner;
	glm::vec3 highestCorner;

	std::vector<int> occupants;
	//int* occupants;
	//std::size_t occupantsSize;

	//OctTree** children;
	std::vector<OctTree*> children;
	
	OctTree()
	{
		glm::vec3 setLowestCorner(std::numeric_limits<float>::infinity());
		glm::vec3 setHighestCorner(-std::numeric_limits<float>::infinity());

		for (int i = 0; i < allParticlesSize / sizeof(Particle); i++)
		{
			occupants.push_back(i);

			setLowestCorner = glm::min(setLowestCorner, allParticles[i].position);
			setHighestCorner = glm::max(setHighestCorner, allParticles[i].position);
		}

		float largestDifference = std::max(setHighestCorner.x - setLowestCorner.x, std::max(setHighestCorner.y - setLowestCorner.y, setHighestCorner.z - setLowestCorner.z));
		setHighestCorner.x = setLowestCorner.x + largestDifference + 0.0001f;
		setHighestCorner.y = setLowestCorner.y + largestDifference + 0.0001f;
		setHighestCorner.z = setLowestCorner.z + largestDifference + 0.0001f;

		lowestCorner = setLowestCorner;
		highestCorner = setHighestCorner;

		std::pair<float, glm::vec3> childMassPosition = createTree();
	}

	OctTree(std::vector<int> initOccupants, glm::vec3 initLowestCorner, glm::vec3 initHighestCorner)
	{
		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;

		//std::cout << "new node created" << std::endl;
	}
	
	std::pair<float, glm::vec3> createTree()
	{
		//std::cout << "createTree begin" << std::endl;
		//std::cout << occupants.size() << std::endl;
		if (occupants.size() > maxChildren)
		{
			//std::cout << "coondition" << std::endl;
			std::vector<int> HHH;
			std::vector<int> HHL;
			std::vector<int> HLH;
			std::vector<int> LHH;
			std::vector<int> HLL;
			std::vector<int> LLH;
			std::vector<int> LHL;
			std::vector<int> LLL;

			float l = (highestCorner.x - lowestCorner.x)/2.0f;
			float middleX = lowestCorner.x + l;
			float middleY = lowestCorner.y + l;
			float middleZ = lowestCorner.z + l;

			for (int i = 0; i < occupants.size(); i++)
			{
				int index = occupants[i];

				if      (allParticles[index].position.x >= middleX && allParticles[index].position.y >= middleY && allParticles[index].position.z >= middleZ)
				{
					HHH.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x >= middleX && allParticles[index].position.y >= middleY && allParticles[index].position.z < middleZ)
				{
					HHL.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x >= middleX && allParticles[index].position.y < middleY && allParticles[index].position.z >= middleZ)
				{
					HLH.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x < middleX && allParticles[index].position.y >= middleY && allParticles[index].position.z >= middleZ)
				{
					LHH.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x >= middleX && allParticles[index].position.y < middleY && allParticles[index].position.z < middleZ)
				{
					HLL.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x < middleX && allParticles[index].position.y < middleY && allParticles[index].position.z >= middleZ)
				{
					LLH.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x < middleX && allParticles[index].position.y >= middleY && allParticles[index].position.z < middleZ)
				{
					LHL.push_back(occupants[i]);
				}
				else if (allParticles[index].position.x < middleX && allParticles[index].position.y < middleY && allParticles[index].position.z < middleZ)
				{
					LLL.push_back(occupants[i]);
				}
				else
				{
					std::cout << "something is wrong in octtree" << std::endl;
				}

				//std::cout << "the " << i << "th value has been added" << std::endl;
			}
			
			if (HHH.size() != 0) { children.push_back(new OctTree(HHH, glm::vec3(middleX,        middleY,        middleZ),        glm::vec3(highestCorner.x, highestCorner.y, highestCorner.z))); }
			if (HHL.size() != 0) { children.push_back(new OctTree(HHL, glm::vec3(middleX,        middleY,        lowestCorner.z), glm::vec3(highestCorner.x, highestCorner.y, middleZ)        )); }
			if (HLH.size() != 0) { children.push_back(new OctTree(HLH, glm::vec3(middleX,        lowestCorner.y, middleZ),        glm::vec3(highestCorner.x, middleY,         highestCorner.z))); }
			if (LHH.size() != 0) { children.push_back(new OctTree(LHH, glm::vec3(lowestCorner.x, middleY,        middleZ),        glm::vec3(middleX,         highestCorner.y, highestCorner.z))); }
			if (HLL.size() != 0) { children.push_back(new OctTree(HLL, glm::vec3(middleX,        lowestCorner.y, lowestCorner.z), glm::vec3(highestCorner.x, middleY,         middleZ)        )); }
			if (LLH.size() != 0) { children.push_back(new OctTree(LLH, glm::vec3(lowestCorner.x, lowestCorner.y, middleZ),        glm::vec3(middleX,         middleY,         highestCorner.z))); }
			if (LHL.size() != 0) { children.push_back(new OctTree(LHL, glm::vec3(lowestCorner.x, middleY,        lowestCorner.z), glm::vec3(middleX,         highestCorner.y, middleZ)        )); }
			if (LLL.size() != 0) { children.push_back(new OctTree(LLL, glm::vec3(lowestCorner.x, lowestCorner.y, lowestCorner.z), glm::vec3(middleX,         middleY,         middleZ)        )); }

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

			for (OctTree* octTree : children)
			{
				std::pair<float, glm::vec3> childMassPosition = octTree->createTree();
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
				totalMass += allParticles[occupants[i]].mass;
				centreOfMass += allParticles[occupants[i]].mass * allParticles[occupants[i]].position;
			}

			centreOfMass /= totalMass;

			return std::make_pair(totalMass, centreOfMass);
		}
	}

	void getLineSegments(std::vector<LineSegment> &lineSegments, int level, int showLevel)
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

			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  lowestCorner.y,  lowestCorner.z),  glm::vec3(highestCorner.x, lowestCorner.y,  lowestCorner.z),  color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  lowestCorner.y,  lowestCorner.z),  glm::vec3(lowestCorner.x,  highestCorner.y, lowestCorner.z),  color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  lowestCorner.y,  lowestCorner.z),  glm::vec3(lowestCorner.x,  lowestCorner.y,  highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  highestCorner.y, highestCorner.z), glm::vec3(highestCorner.x, highestCorner.y, highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(highestCorner.x, highestCorner.y, lowestCorner.z),  glm::vec3(highestCorner.x, highestCorner.y, highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(highestCorner.x, lowestCorner.y,  highestCorner.z), glm::vec3(highestCorner.x, highestCorner.y, highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  lowestCorner.y,  highestCorner.z), glm::vec3(lowestCorner.x,  highestCorner.y, highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(highestCorner.x, lowestCorner.y,  lowestCorner.z),  glm::vec3(highestCorner.x, highestCorner.y, lowestCorner.z),  color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  highestCorner.y, lowestCorner.z),  glm::vec3(lowestCorner.x,  highestCorner.y, highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  highestCorner.y, lowestCorner.z),  glm::vec3(highestCorner.x, highestCorner.y, lowestCorner.z),  color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(lowestCorner.x,  lowestCorner.y,  highestCorner.z), glm::vec3(highestCorner.x, lowestCorner.y,  highestCorner.z), color, color, level));
			lineSegments.push_back(LineSegment(glm::vec3(highestCorner.x, lowestCorner.y,  lowestCorner.z),  glm::vec3(highestCorner.x, lowestCorner.y,  highestCorner.z), color, color, level));
		}


		for (OctTree* octTree : children)
		{
			octTree->getLineSegments(lineSegments, level + 1, showLevel);
		}
	}

	/*
	static void setInit(std::vector<int>& setOccupants, glm::vec3& setLowestCorner, glm::vec3& setHighestCorner)
	{
		setLowestCorner = glm::vec3(std::numeric_limits<float>::infinity());
		setHighestCorner = glm::vec3(-std::numeric_limits<float>::infinity());

		for (int i = 0; i < allParticlesSize / sizeof(Particle); i++)
		{
			setOccupants.push_back(i);

			setLowestCorner = glm::min(setLowestCorner, allParticles[i].position);
			setHighestCorner = glm::max(setHighestCorner, allParticles[i].position);
		}

		float largestDifference = std::max(setHighestCorner.x - setLowestCorner.x, std::max(setHighestCorner.y - setLowestCorner.y, setHighestCorner.z - setLowestCorner.z));
		setHighestCorner.x = setLowestCorner.x + largestDifference + 0.0001f;
		setHighestCorner.y = setLowestCorner.y + largestDifference + 0.0001f;
		setHighestCorner.z = setLowestCorner.z + largestDifference + 0.0001f;
	}
	*/

	~OctTree()
	{

	}
	
private:
};

int OctTree::maxChildren = 1;
Particle* OctTree::allParticles = nullptr;
std::size_t OctTree::allParticlesSize = 0;

#endif