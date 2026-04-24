#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <limits>
#include <utility>
#include <Fastor/Fastor.h>

struct Policy_MP
{
	struct Node_Summary
	{
		glm::dvec2 COM{ 0.0 };

		double M0{0.0};
		Fastor::Tensor<double, 2, 2> M2{ { 0.0, 0.0 }, { 0.0, 0.0 } };
	};

	template <typename T, typename QuadTreeType>
	static inline std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> compute_node
	(
		std::vector<T>& allParticles,
		int maxChildren,
		Node_Summary& summary,
		const std::vector<QuadTreeType*>& children
	)
	{
		for (auto* child : children)
		{
			std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> childPositionMassDiQuad = child->createTree(allParticles, maxChildren);
			summary.M0 += std::get<1>(childPositionMassDiQuad);
			summary.COM += std::get<1>(childPositionMassDiQuad) * std::get<0>(childPositionMassDiQuad);
		}

		if (summary.M0 != 0.0)
			summary.COM /= summary.M0;

		for (auto* child : children)
		{
			// calculate moment as though the child node was a point
			glm::dvec2 relativeCoord = child->summary.COM - summary.COM;

			Fastor::Tensor<double, 2, 2> outer_product;
			outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
			outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
			outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
			outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
			summary.M2 += child->summary.M0 * outer_product;

			// add moment of child node
			summary.M2 += child->summary.M2;
		}

		return std::make_tuple(summary.COM, summary.M0, summary.M2);
	}

	template <typename T>
	static inline std::tuple<glm::dvec2, double, Fastor::Tensor<double, 2, 2>> compute_leaf
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

		for (int i = 0; i < occupants.size(); i++)
		{
			glm::dvec2 relativeCoord = allParticles[occupants[i]].position - summary.COM;

			Fastor::Tensor<double, 2, 2> outer_product;
			outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
			outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
			outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
			outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
			summary.M2 += outer_product;
		}

		return std::make_tuple(summary.COM, summary.M0, summary.M2);
	}

	// interactions----------------------------------------------------------

	template <typename T, typename QuadTreeType>
	static inline void PN_interaction(double& total, T& sinkPoint, QuadTreeType* sourceNode)
	{
		glm::dvec2 R = sinkPoint.position - sourceNode->summary.COM;
		double sq_r = R.x * R.x + R.y * R.y;
		double rS = 1.0 + sq_r;
		
		double D1 = 1.0 / (rS * rS);
		double D2 = -4.0 / (rS * rS * rS);
		double D3 = 24.0 / (rS * rS * rS * rS);
		total += sourceNode->summary.M0 / rS;
		
		double MB0 = sourceNode->summary.M0;
		Fastor::Tensor<double, 2, 2> MB2 = sourceNode->summary.M2;
		Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;
		
		double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
		double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
		double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
		double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
		
		Fastor::Tensor<double, 2> C1 =
		{
		    MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
		    MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
		};
		
		sinkPoint.derivative += glm::dvec2(C1(0), C1(1));
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
			Policy_MP::traverseMP(total, points, points[i], root, theta);
		}
	}

	template <typename T, typename QuadTreeType>
	static inline void traverseMP(double& total, std::vector<T>& allParticles, T& point, QuadTreeType* node, double theta)
	{
		glm::dvec2 diff = point.position - node->summary.COM;

		if (node->BBlength / glm::length(diff) < theta)// && (glm::any(glm::lessThan(point.position, node->lowestCorner)) || glm::any(glm::greaterThan(point.position, node->highestCorner))))
		{

			Policy_MP::PN_interaction(total, point, node);

		}
		else if (node->children.size() <= 1)
		{
			for (int i : node->occupants)
			{
				if (&allParticles[i] != &point) // self intersection test
				{

					Policy_MP::PP_interaction(total, point, allParticles[i]);

				}
			}
		}
		else
		{
			for (QuadTreeType* child : node->children)
			{

				Policy_MP::traverseMP(total, allParticles, point, child, theta);

			}
		}
	}
};