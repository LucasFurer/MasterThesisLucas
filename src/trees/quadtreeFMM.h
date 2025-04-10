#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <vector>
#include <iostream>
#include "../particles/embeddedPoint.h"

#include <Fastor/Fastor.h>

template <typename T>
class QuadTreeFMM
{
public:
	int maxChildren;
	std::vector<T>* allParticles;

	glm::vec2 centreOfMass = glm::vec2(0.0f);

	float totalMass = 0.0f;
	glm::vec2 dipole = glm::vec2(0.0f);
	//glm::mat2 quadrupole = glm::mat2(0.0f);
	Fastor::Tensor<float, 2, 2> quadrupole{};

	glm::vec2 dphi = glm::vec2(0.0f);
	Fastor::Tensor<float, 2, 2> dphi2{};
	Fastor::Tensor<float, 2, 2, 2> dphi3{};

	glm::vec2 lowestCorner = glm::vec2(std::numeric_limits<float>::infinity());
	glm::vec2 highestCorner = glm::vec2(-std::numeric_limits<float>::infinity());

	std::vector<int> occupants;

	std::vector<QuadTreeFMM*> children; // maybe change to no a pointer

	QuadTreeFMM(int initMaxChildren, std::vector<T>* initAllParticles)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		occupants.resize(allParticles->size());
		for (int i = 0; i < allParticles->size(); i++)
		{
			occupants[i] = i;

			lowestCorner = glm::min(lowestCorner, (*allParticles)[i].position);
			highestCorner = glm::max(highestCorner, (*allParticles)[i].position);
		}

		float largestDifference = std::max(highestCorner.x - lowestCorner.x, highestCorner.y - lowestCorner.y);
		highestCorner.x = lowestCorner.x + largestDifference + 0.0001f;
		highestCorner.y = lowestCorner.y + largestDifference + 0.0001f;

		std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> childPositionMassDiQuad = createTree();
		//centreOfMass = std::get<0>(childPositionMassDiQuad); this is already set in the method
		//totalMass = std::get<1>(childPositionMassDiQuad);
		//dipole = std::get<2>(childPositionMassDiQuad);
		//quadrupole = std::get<3>(childPositionMassDiQuad);
	}

	QuadTreeFMM(int initMaxChildren, std::vector<T>* initAllParticles, std::vector<int> initOccupants, glm::vec2 initLowestCorner, glm::vec2 initHighestCorner)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;
	}

	std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> createTree()
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

			if (HH.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, HH, glm::vec2(middleX, middleY), glm::vec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, HL, glm::vec2(middleX, lowestCorner.y), glm::vec2(highestCorner.x, middleY))); }
			if (LH.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, LH, glm::vec2(lowestCorner.x, middleY), glm::vec2(middleX, highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, LL, glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(middleX, middleY))); }

			for (QuadTreeFMM* octTree : children)
			{
				std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> childPositionMassDiQuad = octTree->createTree();
				totalMass += std::get<1>(childPositionMassDiQuad);
				centreOfMass += std::get<1>(childPositionMassDiQuad) * std::get<0>(childPositionMassDiQuad);
			}

			centreOfMass /= totalMass;

			for (QuadTreeFMM* octTree : children)
			{
				// calculate moment as though the child node was a point
				glm::vec2 relativeCoord = octTree->centreOfMass - centreOfMass;
				dipole += octTree->totalMass * relativeCoord;

				Fastor::Tensor<float, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += octTree->totalMass * outer_product;
				//quadrupole += octTree->totalMass * glm::outerProduct(relativeCoord, relativeCoord);

				// add moment of child node
				dipole += octTree->dipole;
				quadrupole += octTree->quadrupole;
			}

			return std::make_tuple(centreOfMass, totalMass, dipole, quadrupole);
		}
		else
		{
			for (int i = 0; i < occupants.size(); i++)
			{
				totalMass += 1.0f;
				centreOfMass += 1.0f * (*allParticles)[occupants[i]].position;
			}

			centreOfMass /= totalMass;

			for (int i = 0; i < occupants.size(); i++)
			{
				glm::vec2 relativeCoord = (*allParticles)[occupants[i]].position - centreOfMass;
				dipole += relativeCoord; // * mass which is always 1

				Fastor::Tensor<float, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += outer_product;
				//quadrupole += glm::outerProduct(relativeCoord, relativeCoord); // * mass which is always 1
			}

			return std::make_tuple(centreOfMass, totalMass, dipole, quadrupole);
		}
	}

	
	void applyForces(std::vector<glm::vec2>* forces)
	{
		if (children.size() != 0)
		{
			for (QuadTreeFMM* child : children)
			{
				child->dphi += dphi;
				//child->dphi2 += dphi2;
				//child->dphi3 += dphi3;
				child->applyForces(forces);
			}
		}
		else
		{
			for (int i : occupants)
			{
				/*
				glm::vec2 dxyz = (*allParticles)[i].position - centreOfMass;
				Fastor::Tensor<float, 2> dxyzT { dxyz.x, dxyz.y };

				Fastor::Tensor<float, 2> dphiP { dphi.x, dphi.y };



				dphiP += einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(dxyzT, dphi2);

				dphiP += 0.5f * einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(dxyzT,
					einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(dxyzT, dphi3));
				
				
				//(*forces)[i] += glm::vec2(dphiP(0), dphiP(1)); // might be -glm::vec2(dphiP(0), dphiP(1))
				(*forces)[i] += -glm::vec2(dphiP(0), dphiP(1)); // might be -glm::vec2(dphiP(0), dphiP(1))
				*/

				(*forces)[i] += dphi;

				//(*forces)[i] += dphi;
			}
		}
	}
	

	void getLineSegments(std::vector<LineSegment2D>& lineSegments, int level, int showLevel)
	{
		if (level == showLevel || showLevel == -1)
		{
			glm::vec3 color;

			switch (showLevel) {
			case -1:
				color = glm::vec3(1.0f, 1.0f, 1.0f);
				break;
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


		for (QuadTreeFMM* octTree : children)
		{
			octTree->getLineSegments(lineSegments, level + 1, showLevel);
		}
	}

	~QuadTreeFMM() {}

private:
};


