#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include "../particles/embeddedPoint.h"

template <typename T>
class QuadTreeBarnesHutReverse
{
public:
	int maxChildren;
	std::vector<T>* allParticles;

	float totalMass;
	glm::vec2 centreOfMass;

	glm::vec2 acceleration = glm::vec2(0.0f);

	glm::vec2 lowestCorner;
	glm::vec2 highestCorner;

	std::vector<int> occupants;

	std::vector<QuadTreeBarnesHutReverse*> children; // maybe change to no a pointer

	QuadTreeBarnesHutReverse(int initMaxChildren, std::vector<T>* initAllParticles)
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

	QuadTreeBarnesHutReverse(int initMaxChildren, std::vector<T>* initAllParticles, std::vector<int> initOccupants, glm::vec2 initLowestCorner, glm::vec2 initHighestCorner)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;
	}

	std::pair<float, glm::vec2> createTree()
	{
		if (occupants.size() > maxChildren)
		{
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
			}

			if (HH.size() != 0) { children.push_back(new QuadTreeBarnesHutReverse(maxChildren, allParticles, HH, glm::vec2(middleX, middleY), glm::vec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTreeBarnesHutReverse(maxChildren, allParticles, HL, glm::vec2(middleX, lowestCorner.y), glm::vec2(highestCorner.x, middleY))); }
			if (LH.size() != 0) { children.push_back(new QuadTreeBarnesHutReverse(maxChildren, allParticles, LH, glm::vec2(lowestCorner.x, middleY), glm::vec2(middleX, highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTreeBarnesHutReverse(maxChildren, allParticles, LL, glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(middleX, middleY))); }

			totalMass = 0.0f;
			centreOfMass = glm::vec2(0.0f);
			for (QuadTreeBarnesHutReverse* quadTreeBarnesHutReverse : children)
			{
				std::pair<float, glm::vec2> childMassPosition = quadTreeBarnesHutReverse->createTree();
				totalMass += childMassPosition.first;
				centreOfMass += childMassPosition.first * childMassPosition.second;
			}

			centreOfMass /= totalMass;

			return std::make_pair(totalMass, centreOfMass);
		}
		else
		{
			totalMass = 0.0f;
			centreOfMass = glm::vec2(0.0f);
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
		if (level == showLevel || showLevel == -1)
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

			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(highestCorner.x, lowestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(lowestCorner.x, highestCorner.y), color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(lowestCorner.x, lowestCorner.y),   color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, highestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(lowestCorner.x, highestCorner.y),  color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y),  glm::vec2(lowestCorner.x, highestCorner.y),  color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y),  glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y),   glm::vec2(highestCorner.x, lowestCorner.y),  color, color, level));
			//lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y),  glm::vec2(highestCorner.x, lowestCorner.y),  color, color, level));
		}


		for (QuadTreeBarnesHutReverse* quadTreeBarnesHutReverse : children)
		{
			quadTreeBarnesHutReverse->getLineSegments(lineSegments, level + 1, showLevel);
		}
	}

	~QuadTreeBarnesHutReverse()
	{
		for (QuadTreeBarnesHutReverse* quadTreeBarnesHutReverse : children)
		{
			delete quadTreeBarnesHutReverse;
		}
	}

private:
};


