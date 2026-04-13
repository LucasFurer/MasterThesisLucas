#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <Fastor/Fastor.h>
#include <utility>

template <typename T>
class QuadTreeFMM
{
public:
	int maxChildren;
	std::vector<T>* allParticles;

	glm::dvec2 centreOfMass = glm::dvec2(0.0);

	double totalMass = 0.0;
	Fastor::Tensor<double, 2, 2> quadrupole{};

	Fastor::Tensor<double, 2> C1{};
	Fastor::Tensor<double, 2, 2> C2{};
	Fastor::Tensor<double, 2, 2, 2> C3{};

	glm::dvec2 lowestCorner = glm::dvec2(std::numeric_limits<double>::infinity());
	glm::dvec2 highestCorner = glm::dvec2(-std::numeric_limits<double>::infinity());

	std::vector<int> occupants;

	std::vector<QuadTreeFMM*> children; // maybe change to no a pointer

	QuadTreeFMM() {}

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

		double largestDifference = std::max(highestCorner.x - lowestCorner.x, highestCorner.y - lowestCorner.y);
		highestCorner.x = lowestCorner.x + largestDifference + 0.0001;
		highestCorner.y = lowestCorner.y + largestDifference + 0.0001;

		std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> childPositionMassDiQuad = createTree();
	}

	QuadTreeFMM(int initMaxChildren, std::vector<T>* initAllParticles, std::vector<int>& initOccupants, glm::dvec2 initLowestCorner, glm::vec2 initHighestCorner)
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;
	}

	QuadTreeFMM(const QuadTreeFMM& other) // copy constructor
	{
		maxChildren = other.maxChildren;
		allParticles = other.allParticles;
		centreOfMass = other.centreOfMass;
		totalMass = other.totalMass;
		quadrupole = other.quadrupole;
		lowestCorner = other.lowestCorner;
		highestCorner = other.highestCorner;
		occupants = other.occupants;

		children = other.children;
		other.children.clear();
	}

	QuadTreeFMM& operator=(const QuadTreeFMM& other) // copy assignment operator
	{
		if (this != &other) // self-assignment check
		{
			maxChildren = other.maxChildren;
			allParticles = other.allParticles;
			centreOfMass = other.centreOfMass;
			totalMass = other.totalMass;
			quadrupole = other.quadrupole;
			lowestCorner = other.lowestCorner;
			highestCorner = other.highestCorner;
			occupants = other.occupants;

			for (QuadTreeFMM* child : children) { delete child; }
			children.clear();

			children.reserve(other.children.size());
			for (const QuadTreeFMM* child : other.children) { children.push_back(new QuadTreeFMM(*child)); }
		}
		return *this;
	}

	QuadTreeFMM(QuadTreeFMM&& other) noexcept // move constructor
	{
		maxChildren = other.maxChildren;
		allParticles = other.allParticles;
		centreOfMass = other.centreOfMass;
		totalMass = other.totalMass;
		quadrupole = other.quadrupole;
		lowestCorner = other.lowestCorner;
		highestCorner = other.highestCorner;
		occupants = std::move(other.occupants);

		children = std::move(other.children);
		other.children.clear();
	}

	QuadTreeFMM& operator=(QuadTreeFMM&& other) noexcept // move assignment operator
	{
		if (this != &other) // self-assignment check
		{
			maxChildren = other.maxChildren;
			allParticles = std::move(other.allParticles);
			other.allParticles = nullptr;

			totalMass = other.totalMass;
			centreOfMass = other.centreOfMass;

			lowestCorner = other.lowestCorner;
			highestCorner = other.highestCorner;
			
			occupants = std::move(other.occupants);

			for (QuadTreeFMM* quadTreeFMM : children) { delete quadTreeFMM; }
			children = std::move(other.children);
		}
		return *this;
	}

	~QuadTreeFMM() // destructor
	{
		for (QuadTreeFMM* quadTreeFMM : children)
		{
			delete quadTreeFMM;
		}
	}

	std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> createTree()
	{
		if (occupants.size() > maxChildren)
		{
			std::vector<int> HH;
			std::vector<int> HL;
			std::vector<int> LH;
			std::vector<int> LL;

			double l = (highestCorner.x - lowestCorner.x) / 2.0;
			double middleX = lowestCorner.x + l;
			double middleY = lowestCorner.y + l;

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

			if (HH.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, HH, glm::dvec2(middleX, middleY), glm::dvec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, HL, glm::dvec2(middleX, lowestCorner.y), glm::dvec2(highestCorner.x, middleY))); }
			if (LH.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, LH, glm::dvec2(lowestCorner.x, middleY), glm::dvec2(middleX, highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTreeFMM(maxChildren, allParticles, LL, glm::dvec2(lowestCorner.x, lowestCorner.y), glm::dvec2(middleX, middleY))); }

			for (QuadTreeFMM* octTree : children)
			{
				std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> childPositionMassDiQuad = octTree->createTree();
				totalMass += std::get<1>(childPositionMassDiQuad);
				centreOfMass += std::get<1>(childPositionMassDiQuad) * std::get<0>(childPositionMassDiQuad);
			}

			centreOfMass /= totalMass;

			for (QuadTreeFMM* octTree : children)
			{
				// calculate moment as though the child node was a point
				glm::dvec2 relativeCoord = octTree->centreOfMass - centreOfMass;

				Fastor::Tensor<double, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += octTree->totalMass * outer_product;

				// add moment of child node
				quadrupole += octTree->quadrupole;
			}

			return std::make_tuple(centreOfMass, totalMass, quadrupole);
		}
		else
		{
			for (int i = 0; i < occupants.size(); i++)
			{
				totalMass += 1.0;
				centreOfMass += 1.0 * (*allParticles)[occupants[i]].position;
			}

			centreOfMass /= totalMass;

			for (int i = 0; i < occupants.size(); i++)
			{
				glm::dvec2 relativeCoord = (*allParticles)[occupants[i]].position - centreOfMass;

				Fastor::Tensor<double, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += outer_product;
			}

			return std::make_tuple(centreOfMass, totalMass, quadrupole);
		}
	}

	
	void applyForces(std::vector<T>& points)
	{
		if (children.size() != 0)
		{
			for (QuadTreeFMM* child : children)
			{
				// prework
				glm::dvec2 oldZ = child->centreOfMass;
				glm::dvec2 newZ = centreOfMass;
				Fastor::Tensor<double, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y }; // dhenen
				Fastor::Tensor<double, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				Fastor::Tensor<double, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// translate C^n to new center of child
				Fastor::Tensor<double, 2> newC1 = C1 +
												 Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
												 (1.0 / 2.0) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));

				Fastor::Tensor<double, 2, 2> newC2 = C2 +
					Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3);

				Fastor::Tensor<double, 2, 2, 2> newC3 = C3;

				// add translated C^n to child C^n
				child->C1 += newC1;
				child->C2 += newC2;
				child->C3 += newC3;

				// try to apply forces for the child node
				child->applyForces(points);
			}
		}
		else
		{
			for (int i : occupants)
			{
				// prework
				glm::dvec2 x = (*allParticles)[i].position;
				glm::dvec2 Z0 = centreOfMass;
				Fastor::Tensor<double, 2> diff1 = { x.x - Z0.x, x.y - Z0.y }; // dhenen
				//Fastor::Tensor<double, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				//Fastor::Tensor<double, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// evaluate C^n at occupants position then add to occupant acceleration // might be wrong!!!!!!!!!!!
				Fastor::Tensor<double, 2> acceleration = C1 +
														Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
														//(1.0 / 2.0) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);
														(1.0 / 2.0) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));
					
				points[i].derivative += glm::dvec2(acceleration(0), acceleration(1));
			}
		}
	}

	void divideC()
	{
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
	

	void getNodesBufferData(std::vector<VertexPos2Col3>& nodesBufferData, int level, int showLevel)
	{
		if (level == showLevel || showLevel == -1)
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

			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(lowestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(highestCorner.x, lowestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(lowestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(lowestCorner.x, highestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(lowestCorner.x, highestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(highestCorner.x, highestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(highestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::dvec2(highestCorner.x, highestCorner.y), color));
		}


		for (QuadTreeFMM* child : children)
		{
			child->getNodesBufferData(nodesBufferData, level + 1, showLevel);
		}
	}

private:
};


