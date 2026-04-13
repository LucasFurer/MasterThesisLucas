#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

#include "../particles/embeddedPoint.h"
#include "../trees/cpu/quadtreeNodeFMMiter.h"
#include "../nbodysolvers/cpu/nBodySolverFMM_MORTON.h"

namespace visHelp
{
	EmbeddedPoint rotate(EmbeddedPoint P, float degrees)
	{
		float radiance = 2.0f * std::numbers::pi_v<float> * degrees / 360.0f;
		return EmbeddedPoint
		(
			glm::vec2
			(
				P.position.x * std::cos(radiance) - P.position.y * std::sin(radiance),
				P.position.x * std::sin(radiance) + P.position.y * std::cos(radiance)
			),
			P.label
		);
	}

	double getRME(const std::vector<TsnePoint2D>& points_naive, const std::vector<TsnePoint2D>& points_approx)
	{
		double MRE = 0.0;

		for (int i = 0; i < points_naive.size(); i++)
		{
			TsnePoint2D pointNaive = points_naive[i];
			TsnePoint2D pointApprox = points_approx[i];

			MRE += static_cast<double>(glm::length(pointNaive.derivative - pointApprox.derivative)) / static_cast<double>(glm::length(pointNaive.derivative));
		}

		MRE = MRE / points_naive.size();

		return MRE;
	}

	void vis_PN_interaction(double& total, TsnePoint2D& sinkPoint, NodeFMM_MORTON_2D& sourceNode)
	{
		glm::dvec2 R = sinkPoint.position - sourceNode.centreOfMass;
		double sq_r = R.x * R.x + R.y * R.y;
		double rS = 1.0 + sq_r;

		double D1 = 1.0 / (rS * rS);
		double D2 = -4.0 / (rS * rS * rS);
		double D3 = 24.0 / (rS * rS * rS * rS);
		total += sourceNode.M0 / rS;

		double MB0 = sourceNode.M0;
		Fastor::Tensor<double, 2, 2> MB2 = sourceNode.M2;
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

	/*
	void NodeNodeBH(std::vector<EmbeddedPoint>& clusterA, std::vector<glm::vec2>& clusterAacc, std::vector<EmbeddedPoint>& clusterB, std::vector<glm::vec2>& clusterBacc)
	{
		glm::vec2 clusterAcentreOfMass(0.0f);
		float clusterAtotalMass = 0.0f;
		for (EmbeddedPoint point : clusterA)
		{
			clusterAcentreOfMass += point.position;
			clusterAtotalMass += 1.0f;
		}
		clusterAcentreOfMass /= clusterAtotalMass;

		glm::vec2 clusterBcentreOfMass(0.0f);
		float clusterBtotalMass = 0.0f;
		for (EmbeddedPoint point : clusterB)
		{
			clusterBcentreOfMass += point.position;
			clusterBtotalMass += 1.0f;
		}
		clusterBcentreOfMass /= clusterBtotalMass;

		for (int i = 0; i < clusterA.size(); i++)
		{
			float softening = 1.0f;

			glm::vec2 R = clusterA[i].position - clusterBcentreOfMass;
			float r = glm::length(R);
			float rS = (r * r) + softening;

			float D1 = -1.0f / (rS * rS);

			R = clusterBtotalMass * R * D1;

			clusterAacc[i] += R;
		}

		for (int i = 0; i < clusterB.size(); i++)
		{
			float softening = 1.0f;

			glm::vec2 R = clusterB[i].position - clusterAcentreOfMass;
			float r = glm::length(R);
			float rS = (r * r) + softening;

			float D1 = -1.0f / (rS * rS);

			R = clusterAtotalMass * R * D1;

			clusterBacc[i] += R;
		}
	}

	void NodeNodeNaive(std::vector<EmbeddedPoint>& clusterA, std::vector<glm::vec2>& clusterAacc, std::vector<EmbeddedPoint>& clusterB, std::vector<glm::vec2>& clusterBacc)
	{
		for (int i = 0; i < clusterA.size(); i++)
		{
			clusterAacc[i] = glm::vec2(0.0f);
			for (int j = 0; j < clusterB.size(); j++)
			{
				float softening = 1.0f;

				glm::vec2 R = clusterA[i].position - clusterB[j].position;
				float r = glm::length(R);
				float rS = (r * r) + softening;

				float D1 = -1.0f / (rS * rS);

				Fastor::Tensor<float, 2> C1 =
				{
					R.x * D1,
					R.y * D1
				};

				clusterAacc[i] += glm::vec2(C1(0), C1(1));
			}
		}

		for (int i = 0; i < clusterB.size(); i++)
		{
			clusterBacc[i] = glm::vec2(0.0f);
			for (int j = 0; j < clusterA.size(); j++)
			{
				float softening = 1.0f;

				glm::vec2 R = clusterB[i].position - clusterA[j].position;
				float r = glm::length(R);
				float rS = (r * r) + softening;

				float D1 = -1.0f / (rS * rS);

				Fastor::Tensor<float, 2> C1 =
				{
					R.x * D1,
					R.y * D1
				};

				clusterBacc[i] += glm::vec2(C1(0), C1(1));
			}
		}
	}

	void NodeNodeFMM(std::vector<EmbeddedPoint>& clusterA, std::vector<glm::vec2>& clusterAacc, std::vector<EmbeddedPoint>& clusterB, std::vector<glm::vec2>& clusterBacc)
	{
		QuadTreeNodeFMMiter<EmbeddedPoint> nodeA(3, clusterA);
		QuadTreeNodeFMMiter<EmbeddedPoint> nodeB(3, clusterB);

		std::cout << "quadrupoleA: " << std::endl;
		std::cout << nodeA.quadrupole(0, 0) << ", " << nodeA.quadrupole(1, 0) << std::endl;
		std::cout << nodeA.quadrupole(0, 1) << ", " << nodeA.quadrupole(1, 1) << std::endl;

		std::cout << "quadrupoleB: " << std::endl;
		std::cout << nodeB.quadrupole(0, 0) << ", " << nodeB.quadrupole(1, 0) << std::endl;
		std::cout << nodeB.quadrupole(0, 1) << ", " << nodeB.quadrupole(1, 1) << std::endl;

		float dummy = 0.0f;
		TSNEFMMiterInteractionKernalNodeNode(&dummy, &nodeA, &nodeB);
		nodeA.divideC();
		nodeB.divideC();

		for (int i = 0; i < clusterA.size(); i++)
		{
			glm::vec2 oldZ = clusterA[i].position;
			glm::vec2 newZ = nodeA.centreOfMass;

			Fastor::Tensor<float, 2> diff = { oldZ.x - newZ.x, oldZ.y - newZ.y };

			Fastor::Tensor<float, 2> newC1 =
				nodeA.C1 +
				Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff, nodeA.C2) +
				(1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff, nodeA.C3));

			clusterAacc[i] = glm::vec2(newC1(0), newC1(1));
		}

		for (int i = 0; i < clusterB.size(); i++)
		{
			glm::vec2 oldZ = clusterB[i].position;
			glm::vec2 newZ = nodeA.centreOfMass;

			Fastor::Tensor<float, 2> diff = { oldZ.x - newZ.x, oldZ.y - newZ.y };

			Fastor::Tensor<float, 2> newC1 =
				nodeA.C1 +
				Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff, nodeA.C2) +
				(1.0f / 2.0f) * Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1>>(diff, Fastor::einsum<Fastor::Index<0>, Fastor::Index<0, 1, 2>>(diff, nodeA.C3));

			clusterBacc[i] = glm::vec2(newC1(0), newC1(1));
		}
	}
	*/
}