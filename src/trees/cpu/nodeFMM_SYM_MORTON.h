#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <Fastor/Fastor.h>
#include <string>
#include <cstddef>
#include <array>

class Node_FMM_SYM_MORTON_2D
{
public:
	glm::dvec2 BB_cen{0.0};
	double BB_len{0.0};

	glm::dvec2 COM{0.0};

	double M0{0.0};
	Fastor::Tensor<double, 2, 2> M2{ {0.0, 0.0}, {0.0, 0.0} };

	Fastor::Tensor<double, 2> C1{ 0.0, 0.0 };
	Fastor::Tensor<double, 2, 2> C2{ {0.0, 0.0}, {0.0, 0.0} };
	Fastor::Tensor<double, 2, 2, 2> C3{ {{0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}} };

	std::size_t first_PI{0u}; // first_particle_index + particle_index_amount is index of last particle
	std::size_t PI_amount{0u}; // amount of particles in node
	std::array<Node_FMM_SYM_MORTON_2D*, 4> sub_nodes{};
	//Node_FMM_SYM_MORTON_2D* sub_node0{nullptr};
	//Node_FMM_SYM_MORTON_2D* sub_node1{nullptr};
	//Node_FMM_SYM_MORTON_2D* sub_node2{nullptr};
	//Node_FMM_SYM_MORTON_2D* sub_node3{nullptr};
	uint8_t sub_nodes_size{0u};

	Node_FMM_SYM_MORTON_2D() = default;

	~Node_FMM_SYM_MORTON_2D()
	{
		for (size_t i = 0; i < sub_nodes_size; i++)
		{
			delete sub_nodes[i];
			sub_nodes[i] = nullptr;
		}
	}

	Node_FMM_SYM_MORTON_2D(const Node_FMM_SYM_MORTON_2D& other)
		: 
		BB_cen(other.BB_cen),
		BB_len(other.BB_len),
		COM(other.COM),
		M0(other.M0), 
		M2(other.M2),
		C1(other.C1), 
		C2(other.C2), 
		C3(other.C3),
		first_PI(other.first_PI),
		PI_amount(other.PI_amount),
		sub_nodes_size(other.sub_nodes_size)
	{
		for (std::size_t i = 0; i < 4; i++) 
		{
			if (other.sub_nodes[i])
			{
				sub_nodes[i] = new Node_FMM_SYM_MORTON_2D(*other.sub_nodes[i]);
			}
			else 
			{
				sub_nodes[i] = nullptr;
			}
		}
	}

	Node_FMM_SYM_MORTON_2D& operator=(const Node_FMM_SYM_MORTON_2D& other) 
	{
		if (this != &other) 
		{
			for (Node_FMM_SYM_MORTON_2D* child : sub_nodes)
			{
				delete child;
			}

			BB_cen = other.BB_cen;
			BB_len = other.BB_len;
			COM = other.COM;
			M0 = other.M0;
			M2 = other.M2;
			C1 = other.C1;
			C2 = other.C2;
			C3 = other.C3;
			first_PI = other.first_PI;
			PI_amount = other.PI_amount;
			sub_nodes_size = other.sub_nodes_size;

			for (std::size_t i = 0; i < 4; ++i) 
			{
				sub_nodes[i] = other.sub_nodes[i] ? new Node_FMM_SYM_MORTON_2D(*other.sub_nodes[i])	: nullptr;
			}
		}
		return *this;
	}
};