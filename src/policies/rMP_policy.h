#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <limits>
#include <utility>
#include <Fastor/Fastor.h>

struct Policy_rMP
{
	struct Node_Summary
	{
		glm::dvec2 COM{ 0.0 };

		double M0{ 0.0 };

		Fastor::Tensor<double, 2> C1{};
		Fastor::Tensor<double, 2, 2> C2{};
		Fastor::Tensor<double, 2, 2, 2> C3{};
	};

	template <typename T, typename QuadTreeType>
	static inline std::pair<double, glm::vec2> compute_node
	(
		std::vector<T>& allParticles,
		int maxChildren,
		Node_Summary& summary,
		const std::vector<QuadTreeType*>& children
	)
	{
		for (auto* child : children)
		{
			std::pair<double, glm::dvec2> childMassPosition = child->createTree(allParticles, maxChildren);
			summary.M0 += childMassPosition.first;
			summary.COM += childMassPosition.first * childMassPosition.second;
		}

		if (summary.M0 != 0.0)
			summary.COM /= summary.M0;

		return std::make_pair(summary.M0, summary.COM);
	}

	template <typename T>
	static inline std::pair<double, glm::vec2> compute_leaf
	(
		std::vector<T>& allParticles,
		int maxChildren,
		Node_Summary& summary,
		const std::vector<int>& occupants
	)
	{
		for (int i = 0; i < occupants.size(); i++)
		{
			summary.M0 += 1.0;
			summary.COM += allParticles[occupants[i]].position;
		}

		if (summary.M0 != 0.0)
			summary.COM /= summary.M0;

		return std::make_pair(summary.M0, summary.COM);
	}

	// interactions----------------------------------------------------------

	template <typename T, typename QuadTreeType>
	static inline void NP_interaction(double& total, QuadTreeType* sinkNode, T& sourcePoint)
	{
		glm::dvec2 R = sinkNode->summary.COM - sourcePoint.position;
		double sq_r = R.x * R.x + R.y * R.y;
		double rS = 1.0 + sq_r;

		double D1 = 1.0 / (rS * rS);
		double D2 = -4.0 / (rS * rS * rS);
		double D3 = 24.0 / (rS * rS * rS * rS);
		total += sinkNode->summary.M0 / rS;

		Fastor::Tensor<double, 2> C1 =
		{
			(R.x * D1),
			(R.y * D1)
		};

		Fastor::Tensor<double, 2, 2> C2 =
		{
			{
				(D1 + R.x * R.x * D2),
				(R.x * R.y * D2)
			},
			{
				(R.y * R.x * D2),
				(D1 + R.y * R.y * D2)
			}
		};

		Fastor::Tensor<double, 2, 2, 2> C3 =
		{
			{
				{
					((R.x + R.x + R.x) * D2 + R.x * R.x * R.x * D3), // i = 0, j = 0, k = 0
					((R.y) * D2 + R.x * R.x * R.y * D3)  // i = 0, j = 0, k = 1
				},
				{
					((R.y) * D2 + R.x * R.y * R.x * D3), // i = 0, j = 1, k = 0
					((R.x) * D2 + R.x * R.y * R.y * D3)  // i = 0, j = 1, k = 1
				}
			},
			{
				{
					((R.y) * D2 + R.y * R.x * R.x * D3), // i = 1, j = 0, k = 0
					((R.x) * D2 + R.y * R.x * R.y * D3)  // i = 1, j = 0, k = 1
				},
				{
					((R.x) * D2 + R.y * R.y * R.x * D3), // i = 1, j = 1, k = 0
					((R.y + R.y + R.y) * D2 + R.y * R.y * R.y * D3)  // i = 1, j = 1, k = 1
				}
			}
		};

		sinkNode->summary.C1 += C1;
		sinkNode->summary.C2 += C2;
		sinkNode->summary.C3 += C3;
	}

	template <typename T>
	static inline void PP_interaction(double& total, T& sinkPoint, T& sourcePoint)
	{
		glm::dvec2 diff = sinkPoint.position - sourcePoint.position;
		double sq_dist = diff.x * diff.x + diff.y * diff.y;

		double forceDecay = 1.0 / (1.0 + sq_dist);
		total += forceDecay;

		sinkPoint.derivative += forceDecay * forceDecay * diff;
	}

	// tree traversal----------------------------------------------------------

	template <typename T, typename QuadTreeType>
	static inline void traverse_tree
	(
		double& total,
		std::vector<T>& points,
		QuadTreeType* root,
		double theta
	)
	{
		for (int i = 0; i < points.size(); i++)
		{
			Policy_rMP::traverse_rMP(total, points, root, points[i], theta);
		}

		Policy_rMP::cascadeValues(points, root);
	}

	template <typename T, typename QuadTreeType>
	static inline void traverse_rMP(double& total, std::vector<T>& allParticles, QuadTreeType* node, T& point, double theta)
	{
		glm::dvec2 diff = point.position - node->summary.COM;


		if (node->BBlength / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
		{

			Policy_rMP::NP_interaction(total, node, point);

		}
		else if (node->children.size() <= 1)
		{
			for (int i : node->occupants)
			{
				if (&allParticles[i] != &point)
				{

					Policy_rMP::PP_interaction(total, allParticles[i], point);

				}
			}
		}
		else
		{
			for (QuadTreeType* child : node->children)
			{

				Policy_rMP::traverse_rMP(total, allParticles, child, point, theta);

			}
		}
	}

	template <typename T, typename QuadTreeType>
	static inline void cascadeValues(std::vector<T>& points, QuadTreeType* node)
	{
		if (node->children.size() != 0)
		{
			for (QuadTreeType* child : node->children)
			{
				// prework
				glm::dvec2 oldZ = child->summary.COM;
				glm::dvec2 newZ = node->summary.COM;
				Fastor::Tensor<double, 2> diff1 = { oldZ.x - newZ.x, oldZ.y - newZ.y }; // dhenen
				//Fastor::Tensor<double, 2> diff1 = { newZ.x - oldZ.x, newZ.y - oldZ.y }; // gadget4
				Fastor::Tensor<double, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				Fastor::Tensor<double, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// translate C^n to new center of child

				//Fastor::Tensor<double, 2> newC1 = C1 +
				//	einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, C2) +
				//	(1.0 / 2.0) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);

				Fastor::Tensor<double, 2> newC1 = node->summary.C1 +
					Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node->summary.C2) +
					(1.0 / 2.0) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node->summary.C3));

				Fastor::Tensor<double, 2, 2> newC2 = node->summary.C2 +
					Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node->summary.C3);

				Fastor::Tensor<double, 2, 2, 2> newC3 = node->summary.C3;

				// add translated C^n to child C^n
				child->summary.C1 += newC1;
				child->summary.C2 += newC2;
				child->summary.C3 += newC3;


				// try to apply forces for the child node
				Policy_rMP::cascadeValues(points, child);
				//child->applyForces(points);
			}
		}
		else
		{
			for (int i : node->occupants)
			{
				// prework
				glm::dvec2 x = points[i].position;
				glm::dvec2 Z0 = node->summary.COM;
				Fastor::Tensor<double, 2> diff1 = { x.x - Z0.x, x.y - Z0.y }; // dhenen
				//Fastor::Tensor<double, 2> diff1 = { Z0.x - x.x, Z0.y - x.y }; // gadget4
				Fastor::Tensor<double, 2, 2> diff2 = Fastor::outer(diff1, diff1);
				Fastor::Tensor<double, 2, 2, 2> diff3 = Fastor::outer(diff2, diff1);

				// evaluate C^n at occupants position then add to occupant acceleration // might be wrong!!!!!!!!!!!
				Fastor::Tensor<double, 2> acceleration = node->summary.C1 +
					Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, node->summary.C2) +
					//(1.0 / 2.0) * einsum<Fastor::Index<0, 1>, Fastor::Index<0, 1, 2>>(diff2, C3);
					(1.0 / 2.0) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff1, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff1, node->summary.C3));

				points[i].derivative += glm::dvec2(acceleration(0), acceleration(1));
			}
		}
	}
};