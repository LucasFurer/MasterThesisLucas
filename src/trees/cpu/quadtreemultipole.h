#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <Fastor/Fastor.h>
#include <utility>

template <typename T>
class QuadTreeMultiPole
{
public:
	int maxChildren;
	std::vector<T>* allParticles;

	glm::dvec2 centreOfMass;

	double totalMass;
	glm::dvec2 dipole;
	Fastor::Tensor<double, 2, 2> quadrupole;

	glm::dvec2 lowestCorner;
	glm::dvec2 highestCorner;

	std::vector<int> occupants;

	std::vector<QuadTreeMultiPole*> children; // maybe change to no a pointer

	QuadTreeMultiPole() {} // empty constructor

	QuadTreeMultiPole(int initMaxChildren, std::vector<T>* initAllParticles) // root constructor
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		glm::dvec2 setLowestCorner(std::numeric_limits<double>::infinity());
		glm::dvec2 setHighestCorner(-std::numeric_limits<double>::infinity());

		for (int i = 0; i < allParticles->size(); i++)
		{
			occupants.push_back(i);

			setLowestCorner = glm::min(setLowestCorner, (*allParticles)[i].position);
			setHighestCorner = glm::max(setHighestCorner, (*allParticles)[i].position);
		}

		double largestDifference = std::max(setHighestCorner.x - setLowestCorner.x, setHighestCorner.y - setLowestCorner.y);
		setHighestCorner.x = setLowestCorner.x + largestDifference + 0.0001;
		setHighestCorner.y = setLowestCorner.y + largestDifference + 0.0001;

		lowestCorner = setLowestCorner;
		highestCorner = setHighestCorner;

		std::tuple<glm::dvec2, double, glm::dvec2, Fastor::Tensor<double, 2, 2>> childPositionMassDiQuad = createTree();
	}

	QuadTreeMultiPole(int initMaxChildren, std::vector<T>* initAllParticles, std::vector<int>& initOccupants, glm::dvec2 initLowestCorner, glm::dvec2 initHighestCorner) // secondary constructor
	{
		maxChildren = initMaxChildren;
		allParticles = initAllParticles;

		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		occupants = initOccupants;
	}

	QuadTreeMultiPole(const QuadTreeMultiPole& other) // copy constructor
	{
		maxChildren = other.maxChildren;
		allParticles = other.allParticles;
		centreOfMass = other.centreOfMass;
		totalMass = other.totalMass;
		dipole = other.dipole;
		quadrupole = other.quadrupole;
		lowestCorner = other.lowestCorner;
		highestCorner = other.highestCorner;
		occupants = other.occupants;

		children = other.children;
		other.children.clear();
	}

	QuadTreeMultiPole& operator=(const QuadTreeMultiPole& other) // copy assignment operator
	{
		if (this != &other) // self-assignment check
		{
			maxChildren = other.maxChildren;
			allParticles = other.allParticles;
			centreOfMass = other.centreOfMass;
			totalMass = other.totalMass;
			dipole = other.dipole;
			quadrupole = other.quadrupole;
			lowestCorner = other.lowestCorner;
			highestCorner = other.highestCorner;
			occupants = other.occupants;

			for (QuadTreeMultiPole* child : children)
			{
				delete child;
			}
			children.clear();

			children.reserve(other.children.size());
			for (const QuadTreeMultiPole* child : other.children)
			{
				children.push_back(new QuadTreeMultiPole(*child));
			}
		}
		return *this;
	}

	QuadTreeMultiPole(QuadTreeMultiPole&& other) noexcept // move constructor
	{
		maxChildren = other.maxChildren;
		allParticles = other.allParticles;
		other.allParticles = nullptr;
		centreOfMass = other.centreOfMass;
		totalMass = other.totalMass;
		dipole = other.dipole;
		quadrupole = other.quadrupole;
		lowestCorner = other.lowestCorner;
		highestCorner = other.highestCorner;
		occupants = std::move(other.occupants);

		children = std::move(other.children);
		other.children.clear();
	}

	QuadTreeMultiPole& operator=(QuadTreeMultiPole&& other) noexcept // move assignment operator
	{
		if (this != &other) // self-assignment check
		{
			maxChildren = other.maxChildren;
			//allParticles = std::move(other.allParticles);
			allParticles = other.allParticles;
			other.allParticles = nullptr;

			centreOfMass = other.centreOfMass;
			totalMass = other.totalMass;

			dipole = other.dipole;
			quadrupole = other.quadrupole;

			lowestCorner = other.lowestCorner;
			highestCorner = other.highestCorner;

			occupants = std::move(other.occupants);

			for (QuadTreeMultiPole* quadTreeBHMP : children) { delete quadTreeBHMP; }
			children = std::move(other.children);
		}
		return *this;
	}

	~QuadTreeMultiPole() // destructor
	{
		for (QuadTreeMultiPole* quadTreeBHMP : children)
		{
			delete quadTreeBHMP;
		}
	}

	std::tuple<glm::dvec2, double, glm::dvec2, Fastor::Tensor<double, 2, 2>> createTree()
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

			if (HH.size() != 0) { children.push_back(new QuadTreeMultiPole(maxChildren, allParticles, HH, glm::dvec2(middleX, middleY), glm::dvec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTreeMultiPole(maxChildren, allParticles, HL, glm::dvec2(middleX, lowestCorner.y), glm::dvec2(highestCorner.x, middleY))); }
			if (LH.size() != 0) { children.push_back(new QuadTreeMultiPole(maxChildren, allParticles, LH, glm::dvec2(lowestCorner.x, middleY), glm::dvec2(middleX, highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTreeMultiPole(maxChildren, allParticles, LL, glm::dvec2(lowestCorner.x, lowestCorner.y), glm::dvec2(middleX, middleY))); }
			
			totalMass = 0.0;
			centreOfMass = glm::dvec2(0.0);
			for (QuadTreeMultiPole* octTree : children)
			{
				std::tuple<glm::dvec2, double, glm::dvec2, Fastor::Tensor<double, 2, 2>> childPositionMassDiQuad = octTree->createTree();
				totalMass += std::get<1>(childPositionMassDiQuad);
				//totalMass += childPositionMassDiQuad.first;
				centreOfMass += std::get<1>(childPositionMassDiQuad) * std::get<0>(childPositionMassDiQuad);
				//centreOfMass += childMassPosition.first * childMassPosition.second;
			}

			centreOfMass /= totalMass;

			quadrupole = Fastor::Tensor<double, 2, 2> {
									 { 0.0, 0.0 },
									 { 0.0, 0.0 }};
			dipole = glm::dvec2(0.0);
			for (QuadTreeMultiPole* octTree : children)
			{
				
				// calculate moment as though the child node was a point
				glm::dvec2 relativeCoord = octTree->centreOfMass - centreOfMass;
				dipole += octTree->totalMass * relativeCoord;

				Fastor::Tensor<double, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += octTree->totalMass * outer_product;
				//quadrupole += octTree->totalMass * glm::outerProduct(relativeCoord, relativeCoord);
				//auto C = einsum<Index<i,j>, Index<i>, Index<j>>(A, B);

				// add moment of child node
				dipole += octTree->dipole;
				quadrupole += octTree->quadrupole;
				

				/*
				//double d_m = d->m; this is child mass
				
				//double qx = d->mx - node->mx; this is relative coord
				//double qy = d->my - node->my;
				//double qz = d->mz - node->mz;
				glm::dvec2 relativeCoord = octTree->centreOfMass - centreOfMass;

				//double qr2 = qx*qx + qy*qy + qz*qz;
				double qr2 = relativeCoord.x * relativeCoord.x + relativeCoord.y * relativeCoord.y;
				//node->mxx += d->mxx + d_m*(3.*qx*qx - qr2);
				quadrupole[0][0] += octTree->quadrupole[0][0] + octTree->totalMass * (3.0f * relativeCoord.x * relativeCoord.x - qr2);
				//node->mxy += d->mxy + d_m * 3. * qx * qy;
				quadrupole[1][0] += octTree->quadrupole[1][0] + octTree->totalMass * (3.0f * relativeCoord.x * relativeCoord.y);
				//node->mxz += d->mxz + d_m * 3. * qx * qz;
				//node->myy += d->myy + d_m * (3. * qy * qy - qr2);
				quadrupole[1][1] += octTree->quadrupole[1][1] + octTree->totalMass * (3.0f * relativeCoord.y * relativeCoord.y - qr2);
				//node->myz += d->myz + d_m * 3. * qy * qz;
				*/
			}

			//node->mzz = -node->mxx - node->myy;
			 
			
			//if ()
			//std::cout << glm::to_string(quadrupole) << std::endl;
			//dipole /= totalMass;
			//quadrupole /= totalMass;

			return std::make_tuple(centreOfMass, totalMass, dipole, quadrupole);
		}
		else
		{
			totalMass = 0.0;
			centreOfMass = glm::dvec2(0.0);
			//std::cout << "leaf" << std::endl;
			for (int i = 0; i < occupants.size(); i++)
			{
				//totalMass += allParticles[occupants[i]].mass;
				totalMass += 1.0;
				//centreOfMass += allParticles[occupants[i]].mass * allParticles[occupants[i]].position;
				centreOfMass += 1.0 * (*allParticles)[occupants[i]].position;
			}

			centreOfMass /= totalMass;

			quadrupole = Fastor::Tensor<double, 2, 2> {
									 { 0.0, 0.0 },
									 { 0.0, 0.0 }};
			dipole = glm::dvec2(0.0);
			for (int i = 0; i < occupants.size(); i++)
			{
				glm::dvec2 relativeCoord = (*allParticles)[occupants[i]].position - centreOfMass;
				dipole += relativeCoord; // * mass which is always 1

				Fastor::Tensor<double, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += outer_product;
				//quadrupole += glm::outerProduct(relativeCoord, relativeCoord); // * mass which is always 1
			}



			//dipole /= totalMass;
			//quadrupole /= totalMass;

			return std::make_tuple(centreOfMass, totalMass, dipole, quadrupole);
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

			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(lowestCorner.x, highestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));

			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, lowestCorner.y), color));
			nodesBufferData.push_back(VertexPos2Col3(glm::vec2(highestCorner.x, highestCorner.y), color));
		}


		for (QuadTreeMultiPole* child : children)
		{
			child->getNodesBufferData(nodesBufferData, level + 1, showLevel);
		}
	}



private:
};


