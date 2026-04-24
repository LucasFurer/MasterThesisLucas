#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>
#include <iostream>
#include <limits>
#include <utility>
#include <Fastor/Fastor.h>

struct Policy_rBH
{
	struct Node_Summary
	{
		glm::dvec2 COM{ 0.0 };

		double M0{ 0.0 };

		Fastor::Tensor<double, 2> C1{ 0.0 };
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
		glm::dvec2 diff = sinkNode->summary.COM - sourcePoint.position;
		double sq_dist = diff.x * diff.x + diff.y * diff.y;

		double forceDecay = 1.0 / (1.0 + sq_dist);
		total += sinkNode->summary.M0 * forceDecay;

		sinkNode->summary.C1 += forceDecay * forceDecay * Fastor::Tensor<double, 2>{diff.x, diff.y};
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
			Policy_rBH::traverse_rBH(total, points, root, points[i], theta);
		}

		Policy_rBH::cascadeValues(points, root, glm::dvec2(0.0));
	}

	template <typename T, typename QuadTreeType>
	static inline void traverse_rBH(double& total, std::vector<T>& allParticles, QuadTreeType* node, T& point, double theta)
	{
		glm::dvec2 diff = point.position - node->summary.COM;


		if (node->BBlength / glm::length(diff) < theta) // && (glm::any(glm::lessThan(particle.position, cubeCentre - l)) || glm::any(glm::greaterThan(particle.position, cubeCentre + l))))
		{

			Policy_rBH::NP_interaction(total, node, point);

		}
		else if (node->children.size() <= 1)
		{
			for (int i : node->occupants)
			{
				if (&allParticles[i] != &point)
				{

					Policy_rBH::PP_interaction(total, allParticles[i], point);

				}
			}
		}
		else
		{
			for (QuadTreeType* child : node->children)
			{

				Policy_rBH::traverse_rBH(total, allParticles, child, point, theta);

			}
		}
	}

	template <typename T, typename QuadTreeType>
	static inline void cascadeValues(std::vector<T>& points, QuadTreeType* node, glm::dvec2 accumulatedVal)
	{
		glm::dvec2 newAccumulatedVal = accumulatedVal + glm::dvec2(node->summary.C1(0), node->summary.C1(1));

		if (node->children.size() <= 1)
		{
			for (int i : node->occupants)
			{

				points[i].derivative += newAccumulatedVal;

			}
		}
		else
		{
			for (QuadTreeType* childQuadTree : node->children)
			{

				Policy_rBH::cascadeValues(points, childQuadTree, newAccumulatedVal);

			}
		}
	}
};