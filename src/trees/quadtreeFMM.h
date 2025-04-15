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
	Fastor::Tensor<float, 2, 2> quadrupole{};

	glm::vec2 tempAccAcc = glm::vec2(0.0f); // delete this one C has been fully implemented
	//float C0 = 0.0f;
	Fastor::Tensor<float, 2> C1{};
	Fastor::Tensor<float, 2, 2> C2{};
	Fastor::Tensor<float, 2, 2, 2> C3{};

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
				//glm::vec2 relativeCoord = centreOfMass - octTree->centreOfMass;
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
				//glm::vec2 relativeCoord = centreOfMass - (*allParticles)[occupants[i]].position;
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
				// prework
				glm::vec2 oldZ = centreOfMass;
				glm::vec2 newZ = child->centreOfMass;
				Fastor::Tensor<float, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y }; // dhenen
				//Fastor::Tensor<float, 2> diff1 = { newZ.x - oldZ.x, newZ.y - oldZ.y }; // gadget4
				Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// translate C^n to new center of child
				/*
				float newC0 = C0 + 
							  einsum<Fastor::Index<0>, Fastor::Index<0>>(diff1, C1)(0) +
							  (1.0f / 2.0f) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1>>(diff2, C2)(0) +
							  (1.0f / 6.0f) * einsum<Fastor::Index<0, 1, 2>, Fastor::Index<0, 1, 2>>(diff3, C3)(0);
				

				Fastor::Tensor<float, 2> newC1 = C1 +
												 einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
												 (1.0f / 2.0f) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);

				//Fastor::Tensor<float, 2> newC1 = C1 +
				//								 einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
				//								 (1.0f / 2.0f) * einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));
				
				Fastor::Tensor<float, 2, 2> newC2 = C2 + 
													einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3);

				Fastor::Tensor<float, 2, 2, 2> newC3 = C3;
				*/

				//Fastor::Tensor<float, 2> newC1 = C1 +
				//	einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
				//	(1.0f / 2.0f) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);

				Fastor::Tensor<float, 2> newC1 = C1 +
												 einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
												 (1.0f / 2.0f) * einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));

				Fastor::Tensor<float, 2, 2> newC2 = C2 +
					einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3);

				Fastor::Tensor<float, 2, 2, 2> newC3 = C3;

				// add translated C^n to child C^n
				child->C1 += newC1;
				//child->C1 += C1;
				child->C2 += newC2;
				//child->C2 += C2;
				child->C3 += newC3;
				//child->C3 += C3;
				child->tempAccAcc += tempAccAcc;

				// try to apply forces for the child node
				child->applyForces(forces);
			}
		}
		else
		{
			for (int i : occupants)
			{
				// prework
				glm::vec2 x = (*allParticles)[i].position;
				glm::vec2 Z0 = centreOfMass;
				Fastor::Tensor<float, 2> diff1 = { x.x - Z0.x, x.y - Z0.y }; // dhenen
				//Fastor::Tensor<float, 2> diff1 = { Z0.x - x.x, Z0.y - x.y }; // gadget4
				Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// evaluate C^n at occupants position then add to occupant acceleration // might be wrong!!!!!!!!!!!
				//Fastor::Tensor<float, 2> temp = einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2);
				//std::cout << "how large is this: " << temp << std::endl;
				Fastor::Tensor<float, 2> acceleration = C1 + 
														einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
														//(1.0f / 2.0f) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);
														(1.0f / 2.0f) * einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));
					
				(*forces)[i] += glm::vec2(acceleration(0), acceleration(1));
				//(*forces)[i] += -glm::vec2(C1(0), C1(1));


				(*forces)[i] += tempAccAcc; // delete this once C^N has been fully implemented 
			}
		}
	}

	void divideC()
	{
		//C0 = C0 / totalMass;
		C1 = C1 / totalMass;
		C2 = C2 / totalMass;
		C3 = C3 / totalMass;

		if (children.size() != 0)
		{
			for (QuadTreeFMM* child : children)
			{
				child->divideC();
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


