#pragma once

template <typename T>
class QuadTreeNodeFMMiter
{
public:
	int id = -1;

	glm::vec2 centreOfMass = glm::vec2(0.0f);

	float totalMass = 0.0f;
	glm::vec2 dipole = glm::vec2(0.0f);
	Fastor::Tensor<float, 2, 2> quadrupole{};

	Fastor::Tensor<float, 2> C1{};
	Fastor::Tensor<float, 2, 2> C2{};
	Fastor::Tensor<float, 2, 2, 2> C3{};

	glm::vec2 lowestCorner = glm::vec2(std::numeric_limits<float>::infinity());
	glm::vec2 highestCorner = glm::vec2(-std::numeric_limits<float>::infinity());

	std::vector<QuadTreeNodeFMMiter*> children; // maybe change to no a pointer

	QuadTreeNodeFMMiter() {}

	QuadTreeNodeFMMiter(int maxChildren, std::vector<T>& allParticles)
	{
		children.resize(allParticles.size());
		for (int i = 0; i < allParticles.size(); i++)
		{
			//std::cout << "all particles position: " << glm::to_string(allParticles[i].position) << std::endl;
			children[i] = new QuadTreeNodeFMMiter(i, allParticles[i].position);
			//std::cout << "all particles position after: " << glm::to_string(children[i]->centreOfMass) << std::endl;

			lowestCorner = glm::min(lowestCorner, allParticles[i].position);
			highestCorner = glm::max(highestCorner, allParticles[i].position);
		}

		float errorRoom = 0.0001f;
		float longestSide = std::max(highestCorner.x - lowestCorner.x, highestCorner.y - lowestCorner.y);
		highestCorner = lowestCorner + longestSide + errorRoom;
		lowestCorner -= 0.0f; //errorRoom;

		//std::cout << "call teh create tree fdunc" << std::endl;
		std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> childPositionMassDiQuad = createTree(maxChildren);
	}
	
	QuadTreeNodeFMMiter(int initId, glm::vec2 initCentreOfMass)
	{
		id = initId;
		centreOfMass = initCentreOfMass;
		totalMass = 1.0f;

		lowestCorner = glm::vec2(initCentreOfMass);
		highestCorner = glm::vec2(initCentreOfMass);
	}

	QuadTreeNodeFMMiter(std::vector<QuadTreeNodeFMMiter*>& initChildren, glm::vec2 initLowestCorner, glm::vec2 initHighestCorner)
	{
		lowestCorner = initLowestCorner;
		highestCorner = initHighestCorner;

		children = std::move(initChildren);
	}

	QuadTreeNodeFMMiter& operator=(QuadTreeNodeFMMiter&& other) // move assignment operator
	{
		//std::cout << "i should be called" << std::endl;
		if (this != &other) // self-assignment check
		{
			id = other.id;

			centreOfMass = other.centreOfMass;

			totalMass = other.totalMass;
			dipole = other.dipole;
			quadrupole = other.quadrupole;
			
			C1 = other.C1;
			C2 = other.C2;
			C3 = other.C3;

			lowestCorner = other.lowestCorner;
			highestCorner = other.highestCorner;

			for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children) { delete quadTreeNodeFMMiter; }
			children = std::move(other.children);
		}
		return *this;
	}

	~QuadTreeNodeFMMiter()
	{
		for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
		{
			delete quadTreeNodeFMMiter;
		}
	}

	std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> createTree(int maxChildren)
	{
		//std::cout << "have called create tree func" << std::endl;

		//std::cout << "size of children1: " << children.size() << std::endl;
		if (children.size() > maxChildren)
		{
			std::vector<QuadTreeNodeFMMiter*> HH;
			std::vector<QuadTreeNodeFMMiter*> HL;
			std::vector<QuadTreeNodeFMMiter*> LH;
			std::vector<QuadTreeNodeFMMiter*> LL;

			float l = (highestCorner.x - lowestCorner.x) / 2.0f;
			float middleX = lowestCorner.x + l;
			float middleY = lowestCorner.y + l;

			for (int i = 0; i < children.size(); i++)
			{
				//std::cout << "child centre of mass: " << glm::to_string(children[i]->centreOfMass) << std::endl;
				//std::cout << "middle x and y: " << middleX << ", " << middleY << std::endl;

				if (children[i]->centreOfMass.x >= middleX && children[i]->centreOfMass.y >= middleY)
				{
					HH.push_back(children[i]);
				}
				else if (children[i]->centreOfMass.x >= middleX && children[i]->centreOfMass.y < middleY)
				{
					HL.push_back(children[i]);
				}
				else if (children[i]->centreOfMass.x < middleX && children[i]->centreOfMass.y >= middleY)
				{
					LH.push_back(children[i]);
				}
				else if (children[i]->centreOfMass.x < middleX && children[i]->centreOfMass.y < middleY)
				{
					LL.push_back(children[i]);
				}
				else
				{
					std::cout << "something is wrong in octtree" << std::endl;
				}
			}

			//std::cout << "size of HH: " << HH.size() << std::endl;
			//std::cout << "size of HL: " << HH.size() << std::endl;
			//std::cout << "size of LH: " << HH.size() << std::endl;
			//std::cout << "size of LL: " << HH.size() << std::endl;

			std::vector<QuadTreeNodeFMMiter*>().swap(children); // empty the children array

			if (HH.size() != 0) { children.push_back(new QuadTreeNodeFMMiter(HH, glm::vec2(middleX,        middleY)       , glm::vec2(highestCorner.x, highestCorner.y))); }
			if (HL.size() != 0) { children.push_back(new QuadTreeNodeFMMiter(HL, glm::vec2(middleX,        lowestCorner.y), glm::vec2(highestCorner.x, middleY))        ); }
			if (LH.size() != 0) { children.push_back(new QuadTreeNodeFMMiter(LH, glm::vec2(lowestCorner.x, middleY)       , glm::vec2(middleX,         highestCorner.y))); }
			if (LL.size() != 0) { children.push_back(new QuadTreeNodeFMMiter(LL, glm::vec2(lowestCorner.x, lowestCorner.y), glm::vec2(middleX,         middleY))        ); }

			//std::cout << "size of children2: " << children.size() << std::endl;

			for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
			{
				std::tuple<glm::vec2, float, glm::vec2, Fastor::Tensor<float, 2, 2>> childPositionMassDiQuad = quadTreeNodeFMMiter->createTree(maxChildren);
				totalMass += std::get<1>(childPositionMassDiQuad);
				centreOfMass += std::get<1>(childPositionMassDiQuad) * std::get<0>(childPositionMassDiQuad);
			}

			centreOfMass /= totalMass;

			for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
			{
				// calculate moment as though the child node was a point
				glm::vec2 relativeCoord = quadTreeNodeFMMiter->centreOfMass - centreOfMass;
				//glm::vec2 relativeCoord = centreOfMass - octTree->centreOfMass;
				dipole += quadTreeNodeFMMiter->totalMass * relativeCoord;

				Fastor::Tensor<float, 2, 2> outer_product;
				outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
				outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
				outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
				outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
				quadrupole += quadTreeNodeFMMiter->totalMass * outer_product;
				//quadrupole += octTree->totalMass * glm::outerProduct(relativeCoord, relativeCoord);

				// add moment of child node
				dipole += quadTreeNodeFMMiter->dipole;
				quadrupole += quadTreeNodeFMMiter->quadrupole;
			}

			return std::make_tuple(centreOfMass, totalMass, dipole, quadrupole);
		}
		else
		{
			for (int i = 0; i < children.size(); i++)
			{
				totalMass += 1.0f;
				centreOfMass += children[i]->centreOfMass; // children[i]->totalMass * children[i]->position
			}

			centreOfMass /= totalMass;

			for (int i = 0; i < children.size(); i++)
			{
				glm::vec2 relativeCoord = children[i]->centreOfMass - centreOfMass;
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
			for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
			{
				// prework
				glm::vec2 oldZ = quadTreeNodeFMMiter->centreOfMass;
				glm::vec2 newZ = centreOfMass;

				Fastor::Tensor<float, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y };
				//Fastor::Tensor<float, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				//Fastor::Tensor<float, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				Fastor::Tensor<float, 2> newC1 =
					C1 +
					Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
					(1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3));

				if (quadTreeNodeFMMiter->id == -1)
				{
					Fastor::Tensor<float, 2, 2> newC2 =
						C2 +
						Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, C3);

					Fastor::Tensor<float, 2, 2, 2> newC3 =
						C3;

					// add translated C^n to child C^n
					quadTreeNodeFMMiter->C1 += newC1;
					quadTreeNodeFMMiter->C2 += newC2;
					quadTreeNodeFMMiter->C3 += newC3;

					// try to apply forces for the child node
					quadTreeNodeFMMiter->applyForces(forces);
				}
				else
				{
					// add translated C^n to child C^n
					quadTreeNodeFMMiter->C1 += newC1;

					(*forces)[quadTreeNodeFMMiter->id] += glm::vec2(quadTreeNodeFMMiter->C1(0), quadTreeNodeFMMiter->C1(1));
				}
			}
		}
	}

	void divideC()
	{
		C1 /= totalMass;
		C2 /= totalMass;
		C3 /= totalMass;

		if (children.size() != 0)
		{
			for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
			{
				quadTreeNodeFMMiter->divideC();
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
			lineSegments.push_back(LineSegment2D(glm::vec2(lowestCorner.x, highestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
			lineSegments.push_back(LineSegment2D(glm::vec2(highestCorner.x, lowestCorner.y), glm::vec2(highestCorner.x, highestCorner.y), color, color, level));
		}


		for (QuadTreeNodeFMMiter* quadTreeNodeFMMiter : children)
		{
			if (quadTreeNodeFMMiter->children.size() != 0)
			{
				quadTreeNodeFMMiter->getLineSegments(lineSegments, level + 1, showLevel);
			}
		}
	}
private:
};


