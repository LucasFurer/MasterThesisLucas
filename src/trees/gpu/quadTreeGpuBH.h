//#pragma once
//
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//
//#include <algorithm>
//#include <vector>
//#include <iostream>
//
//template <typename T>
//class QuadTreeGpuBH
//{
//public:
//	glm::vec2 centreBB;
//	float lengthBB;
//
//	int firstChildIndex; // max of 4 children
//	int childAmount; // firstChildIndex + childAmount - 1 = last child of this node
//
//	int firstParticleIndex;
//	int particleAmount; // firstParticleIndex + particleAmount - 1 = last particle in this node
//
//	float totalMass;
//	glm::vec2 centreOfMass;
//
//	QuadTreeGpuBH() 
//	{
//		centreBB = glm::vec2(0.0f);
//		lengthBB = 0.0f;
//
//		firstChildIndex = 0;
//		firstParticleIndex = 0;
//		particleAmount = 0;
//
//		totalMass = 0.0f;
//		centreOfMass = glm::vec2(0.0f);
//	}
//
//	QuadTreeGpuBH(std::vector<T>& particles, std::vector<int>& indexTracker, std::vector<QuadTreeGpuBH>& nodes, int maxChildren)
//	{
//		glm::vec2 lowestCorner(std::numeric_limits<float>::infinity());
//		glm::vec2 highestCorner(-std::numeric_limits<float>::infinity());
//
//		for (int i = 0; i < allParticles->size(); i++)
//		{
//			lowestCorner = glm::min(lowestCorner, particles[i].position);
//			highestCorner = glm::max(highestCorner, particles[i].position);
//		}
//
//		centreBB = 0.5f * (lowestCorner + highestCorner);
//		lengthBB = glm::compMax(highestCorner - lowestCorner) + 0.0001f; // is + 0.0001f even needed?
//
//
//
//		//std::pair<float, glm::vec2> childMassPosition = createTree();
//
//		firstChildIndex = -1;
//		childAmount = 0;
//		firstParticleIndex = 0;
//		particleAmount = particles.size();
//
//
//		createTree(particles, indexTracker, nodes, maxChildren);
//	}
//
//	QuadTreeGpuBH(int initMaxChildren, std::vector<T>* initAllParticles, std::vector<int>& initOccupants, glm::vec2 initLowestCorner, glm::vec2 initHighestCorner)
//	{
//		maxChildren = initMaxChildren;
//		allParticles = initAllParticles;
//
//		lowestCorner = initLowestCorner;
//		highestCorner = initHighestCorner;
//
//		occupants = initOccupants;
//	}
//
//	~QuadTreeGpuBH() {}
//
//	void createTree(std::vector<T>& particles, std::vector<int>& indexTracker, std::vector<QuadTreeGpuBH>& nodes, int maxChildren);
//	{
//		totalMass = 0.0f;
//		centreOfMass = glm::vec2(0.0f);
//
//		// subdivide into four children if there are more than MAX_CHILDREN in particleAmount
//		if (particleAmount > maxChildren)
//		{
//			return;
//		}
//		else
//		{
//			return;
//		}
//
//	}
//
//	inline void swapParticle(std::vector<T>& particles, std::vector<int>& indexTracker, int i, int j)
//	{
//		indexTracker[particles[i].ID] = j;
//		indexTracker[particles[j].ID] = i;
//
//		std::swap(particles[i], particles[j]);
//	}
//
//	/*
//	std::pair<float, glm::vec2> createTree()
//	{
//		if (occupants.size() > maxChildren)
//		{
//			std::vector<int> HH;
//			std::vector<int> HL;
//			std::vector<int> LH;
//			std::vector<int> LL;
//
//			float l = (highestCorner.x - lowestCorner.x) / 2.0f;
//			float middleX = lowestCorner.x + l;
//			float middleY = lowestCorner.y + l;
//
//			for (int i = 0; i < occupants.size(); i++)
//			{
//				int index = occupants[i];
//
//				if ((*allParticles)[index].position.x >= middleX && (*allParticles)[index].position.y >= middleY)
//				{
//					HH.push_back(occupants[i]);
//				}
//				else if ((*allParticles)[index].position.x >= middleX && (*allParticles)[index].position.y < middleY)
//				{
//					HL.push_back(occupants[i]);
//				}
//				else if ((*allParticles)[index].position.x < middleX && (*allParticles)[index].position.y >= middleY)
//				{
//					LH.push_back(occupants[i]);
//				}
//				else if ((*allParticles)[index].position.x < middleX && (*allParticles)[index].position.y < middleY)
//				{
//					LL.push_back(occupants[i]);
//				}
//				else
//				{
//					std::cout << "something is wrong in octtree" << std::endl;
//				}
//			}
//
//			if (HH.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, HH, glm::vec2(middleX, middleY), glm::vec2(highestCorner.x, highestCorner.y))); }
//			if (HL.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, HL, glm::vec2(middleX, lowestCorner.y), glm::vec2(highestCorner.x, middleY))); }
//			if (LH.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, LH, glm::vec2(lowestCorner.x, middleY), glm::vec2(middleX, highestCorner.y))); }
//			if (LL.size() != 0) { children.push_back(new QuadTree(maxChildren, allParticles, LL, glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(middleX, middleY))); }
//
//			totalMass = 0.0f;
//			centreOfMass = glm::vec2(0.0f);
//			for (QuadTree* quadTree : children)
//			{
//				std::pair<float, glm::vec2> childMassPosition = quadTree->createTree();
//				totalMass += childMassPosition.first;
//				centreOfMass += childMassPosition.first * childMassPosition.second;
//			}
//
//			centreOfMass /= totalMass;
//
//			return std::make_pair(totalMass, centreOfMass);
//		}
//		else
//		{
//			totalMass = 0.0f;
//			centreOfMass = glm::vec2(0.0f);
//			//std::cout << "leaf" << std::endl;
//			for (int i = 0; i < occupants.size(); i++)
//			{
//				//totalMass += allParticles[occupants[i]].mass;
//				totalMass += 1.0f;
//				//centreOfMass += allParticles[occupants[i]].mass * allParticles[occupants[i]].position;
//				centreOfMass += 1.0f * (*allParticles)[occupants[i]].position;
//			}
//
//			centreOfMass /= totalMass;
//
//			return std::make_pair(totalMass, centreOfMass);
//		}
//	}
//	*/
//
//	/*
//	void getLineSegments(std::vector<LineSegment2D>& lineSegments, int level, int showLevel)
//	{
//		if (level == showLevel || showLevel == -1)
//		{
//			glm::vec3 color;
//
//			switch (showLevel) {
//			case 0:
//				color = glm::vec3(1.0f, 0.0f, 0.0f);
//				break;
//			case 1:
//				color = glm::vec3(0.0f, 1.0f, 0.0f);
//				break;
//			case 2:
//				color = glm::vec3(0.0f, 0.0f, 1.0f);
//				break;
//			case 3:
//				color = glm::vec3(1.0f, 1.0f, 0.0f);
//				break;
//			case 4:
//				color = glm::vec3(0.0f, 1.0f, 1.0f);
//				break;
//			case 5:
//				color = glm::vec3(1.0f, 0.0f, 1.0f);
//				break;
//			default:
//				color = glm::vec3(1.0f, 1.0f, 1.0f);
//			}
//
//			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(highestCorner.x, lowestCorner.y), color, color, level));
//			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(lowestCorner.x, highestCorner.y), color, color, level));
//			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
//			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
//		}
//
//
//		for (QuadTree* octTree : children)
//		{
//			octTree->getLineSegments(lineSegments, level + 1, showLevel);
//		}
//	}
//	*/
//
//private:
//};
//
//
