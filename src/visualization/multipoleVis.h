//#pragma once
//
//#include <iostream>
//#include <fstream>
//#include <filesystem>
//#include <vector>
//#include <cmath>
//#include <numbers>
//#include <glm/glm.hpp>
//#include <glm/gtx/color_space.hpp>
//#include <limits>
//
//#include "../particles/embeddedPoint.h"
//#include "../trees/cpu/quadtreeNodeFMMiter.h"
//#include "../nbodysolvers/cpu/nBodySolverFMMiter.h"
//#include "visHelp.h"
//
//namespace MultipoleVis
//{
//	double xmin{ -8.0 };
//	double xmax{ 8.0 };
//	double ymin{ -8.0 };
//	double ymax{ 8.0 };
//
//	std::vector<TsnePoint2D> points = 
//	{
//		TsnePoint2D(glm::dvec2(-1.0, 0.5),  glm::dvec2(0.0), 0, 0, std::uint32_t(0)),
//		TsnePoint2D(glm::dvec2(-1.0, -0.5), glm::dvec2(0.0), 0, 0, std::uint32_t(0)),
//		//TsnePoint2D(glm::dvec2(0.86602540378 - 1.0, 0.0),   glm::dvec2(0.0), 0, 0, std::uint32_t(0))
//		TsnePoint2D(glm::dvec2(2.0, 0.0),   glm::dvec2(0.0), 0, 0, std::uint32_t(0))
//	};
//
//	void create_PPM(std::vector<std::vector<glm::u8vec3>>& pixels, size_t max_color_val)
//	{
//		std::ofstream file(std::filesystem::current_path() / "PPM_files" / "image.ppm", std::ios::binary);
//		if (!file)
//			std::cout << "couldnt open file\n";
//		
//		size_t width = pixels[0].size();
//		size_t height = pixels.size();
//
//		file << "P6\n";
//		file << width << " " << height << "\n";
//		file << max_color_val << "\n";
//		for (size_t y = 0; y < height; y++)
//			file.write(reinterpret_cast<const char*>(pixels[y].data()), width * sizeof(glm::u8vec3));
//		//file.write(reinterpret_cast<const char*>(pixels.data()), width * height * sizeof(glm::u8vec3));
//
//		file.close();
//	}
//
//	void create_vis1()
//	{
//		size_t width = 2000;
//		size_t height = 2000;
//		size_t max_color_val = 255;
//
//		std::vector<std::vector<glm::dvec3>> values(height, std::vector<glm::dvec3>(width, glm::dvec3(0.0)));
//
//		//double max_val{0.0};
//		//double min_val{0.0};
//		double max_r{ 0.0 };
//
//		for (size_t x = 0; x < width; x++) // create vals
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				glm::dvec2 coor = glm::dvec2
//				(
//					xmin + (xmax - xmin) * (static_cast<double>(x) / (static_cast<double>(width - 1))),
//					ymin + (ymax - ymin) * (static_cast<double>(y) / (static_cast<double>(height - 1)))
//				);
//
//				glm::dvec2 centreOfMass(0.0);
//				for (TsnePoint2D point : points)
//				{
//					centreOfMass += point.position;
//				}
//				double M0 = points.size();
//				centreOfMass /= M0;
//
//
//				//Fastor::Tensor<double, 2, 2> M2{ {6.0, 0.0}, {0.0, 0.5} };
//				Fastor::Tensor<double, 2, 2> M2 = { {0.0, 0.0}, {0.0, 0.0} };
//				for (TsnePoint2D point : points)
//				{
//					glm::dvec2 relativeCoord = point.position - centreOfMass;
//
//					Fastor::Tensor<double, 2, 2> outer_product;
//					outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
//					outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
//					outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
//					outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
//					M2 += outer_product;
//				}
//
//				// calculate the field
//				glm::dvec2 field(0.0);
//				double total{ 0.0 };
//
//
//
//
//				if (false)
//				{
//					for (TsnePoint2D point : points)
//					{
//						glm::dvec2 R = coor - point.position;
//						double sq_dist = R.x * R.x + R.y * R.y;
//
//						double forceDecay = 1.0 / (1.0 + sq_dist);
//						total += forceDecay;
//
//						field += forceDecay * forceDecay * R;
//					}
//					//field *= 1.0 / total;
//				}
//				else if (false)
//				{
//					glm::dvec2 R = coor - centreOfMass;
//					double sq_dist = R.x * R.x + R.y * R.y;
//
//					double forceDecay = 1.0 / (1.0 + sq_dist);
//					total += M0 * forceDecay;
//
//					field += M0 * forceDecay * forceDecay * R;
//					//field *= 1.0 / total;
//				}
//				else
//				{
//					glm::dvec2 R = coor - centreOfMass;
//					double sq_r = R.x * R.x + R.y * R.y;
//					double rS = 1.0 + sq_r;
//
//					double D1 = 1.0 / (rS * rS);
//					double D2 = -4.0 / (rS * rS * rS);
//					double D3 = 24.0 / (rS * rS * rS * rS);
//					total += M0 / rS;
//
//					double MB0 = M0;
//					Fastor::Tensor<double, 2, 2> MB2 = M2;
//					Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;
//
//
//					double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//					double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//					double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//					double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//					Fastor::Tensor<double, 2> C1 =
//					{
//						MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//						MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//					};
//
//					field += glm::dvec2(C1(0), C1(1));
//					//field *= 1.0 / total;
//				}
//
//
//
//
//
//
//				
//				max_r = std::max(max_r, glm::length(field));
//				//max_r = 6.43953;
//				//max_r = 0.974278;
//				//max_r = 0.486039;
//				
//				max_r = (0.974278 + 0.486039) / 2.0;
//
//				//max_r = std::max(max_r, glm::length(coor));
//				values[y][x].r = field.x;
//				values[y][x].g = field.y;
//				values[y][x].b = 0.0;
//
//				//min_val = std::min(min_val, std::min(values[y][x].r, std::min(values[y][x].g, values[y][x].b)));
//				//max_val = std::max(max_val, std::max(values[y][x].r, std::max(values[y][x].g, values[y][x].b)));
//			}
//		}
//
//		std::cout << "max_r: " << max_r << std::endl;
//
//		for (size_t x = 0; x < width; x++) // normalize
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				double angle_radians = std::atan2(values[y][x].g, values[y][x].r);
//				angle_radians = angle_radians < 0.0 ? angle_radians + 2.0 * std::numbers::pi : angle_radians;
//				double radius = glm::length(glm::dvec2(values[y][x].r, values[y][x].g)) / max_r;
//				if (radius > 1.0) { radius = 1.0; }
//
//				glm::dvec3 hsv(angle_radians * 360.0 / (2.0 * std::numbers::pi), radius, 1.0);
//				glm::dvec3 rgb = glm::rgbColor(hsv);
//
//				values[y][x].r = rgb.r;
//				values[y][x].g = rgb.g;
//				values[y][x].b = rgb.b;
//			}
//		}
//
//
//
//
//
//
//		std::vector<std::vector<glm::u8vec3>> pixels(height, std::vector<glm::u8vec3>(width, glm::u8vec3(0u)));
//
//		for (size_t x = 0; x < width; x++)
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				//if (values[y][x].r > 1.0) { values[y][x].r = 0.0; }
//				//if (values[y][x].g > 1.0) { values[y][x].g = 0.0; }
//				//if (values[y][x].b > 1.0) { values[y][x].b = 0.0; }
//
//				pixels[y][x].r = static_cast<uint8_t>(values[y][x].r * 255.0);
//				pixels[y][x].g = static_cast<uint8_t>(values[y][x].g * 255.0);
//				pixels[y][x].b = static_cast<uint8_t>(values[y][x].b * 255.0);
//			}
//		}
//
//		//for (TsnePoint2D point : points)
//		//{
//		//	double x_ratio = (point.position.x - xmin) / (xmax - xmin);
//		//	double y_ratio = (point.position.y - ymin) / (ymax - ymin);
//
//		//	double point_radius = 3.0;
//		//	int point_radius_int = static_cast<int>(std::ceil(point_radius));
//		//	int point_coord_x = x_ratio * width;
//		//	int point_coord_y = y_ratio * height;
//		//	
//		//	for (int x = -point_radius_int; x <= point_radius_int; x++)
//		//	{
//		//		for (int y = -point_radius_int; y <= point_radius_int; y++)
//		//		{
//		//			if (point_coord_x + x > 0 && point_coord_x + x < width && point_coord_y + y > 0 && point_coord_y + y < height)
//		//			{
//		//				if (glm::length(glm::dvec2(x, y)) < point_radius)
//		//				{
//		//					pixels[point_coord_y + y][point_coord_x + x].r = 0u;
//		//					pixels[point_coord_y + y][point_coord_x + x].g = 0u;
//		//					pixels[point_coord_y + y][point_coord_x + x].b = 0u;
//		//				}
//		//			}
//		//		}
//		//	}
//		//}
//
//		create_PPM(pixels, max_color_val);
//	}
//
//	void create_vis2()
//	{
//		size_t width = 2000;
//		size_t height = 2000;
//		size_t max_color_val = 255;
//
//		std::vector<std::vector<glm::dvec3>> values(height, std::vector<glm::dvec3>(width, glm::dvec3(0.0)));
//
//		//double max_val{0.0};
//		//double min_val{0.0};
//		double max_r{ 0.0 };
//
//		for (size_t x = 0; x < width; x++) // create vals
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				glm::dvec2 coor = glm::dvec2
//				(
//					xmin + (xmax - xmin) * (static_cast<double>(x) / (static_cast<double>(width - 1))),
//					ymin + (ymax - ymin) * (static_cast<double>(y) / (static_cast<double>(height - 1)))
//				);
//
//				glm::dvec2 centreOfMass(0.0);
//				for (TsnePoint2D point : points)
//				{
//					centreOfMass += point.position;
//				}
//				double M0 = points.size();
//				centreOfMass /= M0;
//
//
//				//Fastor::Tensor<double, 2, 2> M2{ {6.0, 0.0}, {0.0, 0.5} };
//				Fastor::Tensor<double, 2, 2> M2 = { {0.0, 0.0}, {0.0, 0.0} };
//				for (TsnePoint2D point : points)
//				{
//					glm::dvec2 relativeCoord = point.position - centreOfMass;
//
//					Fastor::Tensor<double, 2, 2> outer_product;
//					outer_product(0, 0) = relativeCoord.x * relativeCoord.x;
//					outer_product(0, 1) = relativeCoord.x * relativeCoord.y;
//					outer_product(1, 0) = relativeCoord.y * relativeCoord.x;
//					outer_product(1, 1) = relativeCoord.y * relativeCoord.y;
//					M2 += outer_product;
//				}
//
//				// calculate the field
//				//glm::dvec2 field(0.0);
//				//double total{ 0.0 };
//
//
//
//
//				glm::dvec2 field_naive(0.0);
//				double total_naive{ 0.0 };
//				{
//					for (TsnePoint2D point : points)
//					{
//						glm::dvec2 R = coor - point.position;
//						double sq_dist = R.x * R.x + R.y * R.y;
//
//						double forceDecay = 1.0 / (1.0 + sq_dist);
//						total_naive += forceDecay;
//
//						field_naive += forceDecay * forceDecay * R;
//					}
//				}
//				//field_naive *= 1.0 / total_naive;
//
//
//
//				glm::dvec2 field_BH(0.0);
//				double total_BH{ 0.0 };
//				{
//					glm::dvec2 R = coor - centreOfMass;
//					double sq_dist = R.x * R.x + R.y * R.y;
//
//					double forceDecay = 1.0 / (1.0 + sq_dist);
//					total_BH += M0 * forceDecay;
//
//					field_BH += M0 * forceDecay * forceDecay * R;
//				}
//				//field_BH *= 1.0 / total_BH;
//
//
//
//				glm::dvec2 field_MM(0.0);
//				double total_MM{ 0.0 };
//				{
//					glm::dvec2 R = coor - centreOfMass;
//					double sq_r = R.x * R.x + R.y * R.y;
//					double rS = 1.0 + sq_r;
//
//					double D1 = 1.0 / (rS * rS);
//					double D2 = -4.0 / (rS * rS * rS);
//					double D3 = 24.0 / (rS * rS * rS * rS);
//					total_MM += M0 / rS;
//
//					double MB0 = M0;
//					Fastor::Tensor<double, 2, 2> MB2 = M2;
//					Fastor::Tensor<double, 2, 2> MB2Tilde = (1.0 / MB0) * MB2;
//
//
//					double MB2TildeSum1 = MB2Tilde(0, 0) + MB2Tilde(1, 1);
//					double MB2TildeSum2 = (R.x * R.x * MB2Tilde(0, 0)) + (R.x * R.y * MB2Tilde(0, 1)) + (R.y * R.x * MB2Tilde(1, 0)) + (R.y * R.y * MB2Tilde(1, 1));
//					double MB2TildeSum3i0 = R.x * MB2Tilde(0, 0) + R.y * MB2Tilde(0, 1);
//					double MB2TildeSum3i1 = R.x * MB2Tilde(1, 0) + R.y * MB2Tilde(1, 1);
//
//					Fastor::Tensor<double, 2> C1 =
//					{
//						MB0 * (R.x * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i0)*D2),
//						MB0 * (R.y * (D1 + 0.5 * (MB2TildeSum1)*D2 + 0.5 * (MB2TildeSum2)*D3) + (MB2TildeSum3i1)*D2)
//					};
//
//					field_MM += glm::dvec2(C1(0), C1(1));
//				}
//				//field_MM *= 1.0 / total_MM;
//				
//
//
//
//
//				double BH_error = (glm::length(field_naive - field_BH)) / glm::length(field_naive);
//				double MM_error = (glm::length(field_naive - field_MM)) / glm::length(field_naive);
//
//
//				//double rel_error = BH_error - MM_error;
//				//max_r = std::max(max_r, glm::length(field));
//				//max_r = 1.0;
//
//				//max_r = std::max(max_r, glm::length(coor));
//				values[y][x].r = MM_error;
//				values[y][x].g = MM_error;
//				values[y][x].b = MM_error;
//
//				//min_val = std::min(min_val, std::min(values[y][x].r, std::min(values[y][x].g, values[y][x].b)));
//				//max_val = std::max(max_val, std::max(values[y][x].r, std::max(values[y][x].g, values[y][x].b)));
//			}
//		}
//
//		for (size_t x = 0; x < width; x++) // normalize
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				//double angle_radians = std::atan2(values[y][x].g, values[y][x].r);
//				//angle_radians = angle_radians < 0.0 ? angle_radians + 2.0 * std::numbers::pi : angle_radians;
//				//double radius = glm::length(glm::dvec2(values[y][x].r, values[y][x].g)) / max_r;
//				//if (radius > 1.0) { radius = 0.0; }
//
//				//glm::dvec3 hsv(angle_radians * 360.0 / (2.0 * std::numbers::pi), radius, 1.0);
//				//glm::dvec3 rgb = glm::rgbColor(hsv);
//
//				double cutoff = 0.1;
//				if (values[y][x].r > cutoff)
//				{
//					values[y][x].r = 0.0;
//					values[y][x].g = 0.0;
//					values[y][x].b = 0.0;
//				}
//				else
//				{
//					double val = 1.0 - ((1.0 / cutoff) * values[y][x].r);
//
//					values[y][x].r = 1.0;
//					values[y][x].g = val;
//					values[y][x].b = val;
//				}
//				//else
//				//{
//				//	values[y][x].r = 0.0;
//				//	values[y][x].g = 0.0;
//				//	values[y][x].b = 0.0;
//				//}
//			}
//		}
//
//
//
//
//
//
//		std::vector<std::vector<glm::u8vec3>> pixels(height, std::vector<glm::u8vec3>(width, glm::u8vec3(0u)));
//
//		for (size_t x = 0; x < width; x++)
//		{
//			for (size_t y = 0; y < height; y++)
//			{
//				//if (values[y][x].r > 1.0) { values[y][x].r = 0.0; }
//				//if (values[y][x].g > 1.0) { values[y][x].g = 0.0; }
//				//if (values[y][x].b > 1.0) { values[y][x].b = 0.0; }
//
//				pixels[y][x].r = static_cast<uint8_t>(values[y][x].r * 255.0);
//				pixels[y][x].g = static_cast<uint8_t>(values[y][x].g * 255.0);
//				//pixels[y][x].g = uint8_t{};
//				pixels[y][x].b = static_cast<uint8_t>(values[y][x].b * 255.0);
//				//pixels[y][x].b = uint8_t{};
//			}
//		}
//
//		for (TsnePoint2D point : points)
//		{
//			double x_ratio = (point.position.x - xmin) / (xmax - xmin);
//			double y_ratio = (point.position.y - ymin) / (ymax - ymin);
//
//			double point_radius = 3.0;
//			int point_radius_int = static_cast<int>(std::ceil(point_radius));
//			int point_coord_x = x_ratio * width;
//			int point_coord_y = y_ratio * height;
//
//			for (int x = -point_radius_int; x <= point_radius_int; x++)
//			{
//				for (int y = -point_radius_int; y <= point_radius_int; y++)
//				{
//					if (point_coord_x + x > 0 && point_coord_x + x < width && point_coord_y + y > 0 && point_coord_y + y < height)
//					{
//						if (glm::length(glm::dvec2(x, y)) < point_radius)
//						{
//							//pixels[point_coord_y + y][point_coord_x + x].r = 0u;
//							//pixels[point_coord_y + y][point_coord_x + x].g = 0u;
//							//pixels[point_coord_y + y][point_coord_x + x].b = 0u;
//						}
//					}
//				}
//			}
//
//
//		}
//
//		create_PPM(pixels, max_color_val);
//	}
//
//	/*
//	glm::vec2 clusterOfsett = glm::vec2(10.0f, 0.0f);
//	std::vector<EmbeddedPoint> smallCluster
//	{
//		EmbeddedPoint(glm::vec2(-1.0f,  0.5f), 0),
//		EmbeddedPoint(glm::vec2(-1.0f, -0.5f), 0),
//		EmbeddedPoint(glm::vec2( 1.0f,  0.0f), 0)
//	};
//	std::vector<EmbeddedPoint> smallClusterAngled(smallCluster.size());
//
//	std::vector<EmbeddedPoint> centeredCluster
//	{
//		EmbeddedPoint(glm::vec2(-1.0f,  0.5f), 0),
//		EmbeddedPoint(glm::vec2(-1.0f, -0.5f), 0),
//		EmbeddedPoint(glm::vec2(1.0f,  0.0f), 0)
//	};
//
//	void initMultipoleVisData()
//	{
//		for (int i = 0; i < smallCluster.size(); i++)
//		{
//			smallClusterAngled[i] = visHelp::rotate(smallCluster[i], -45.0f);
//			smallClusterAngled[i].position -= clusterOfsett;
//
//			smallCluster[i].position += clusterOfsett;
//		}
//	}
//
//	void testFMMtoBH()
//	{
//		std::vector<glm::vec2> clusterAaccNaive(3, glm::vec2(0.0f));
//		std::vector<glm::vec2> clusterBaccNaive(3, glm::vec2(0.0f));
//		visHelp::NodeNodeNaive(smallCluster, clusterAaccNaive, smallClusterAngled, clusterBaccNaive);
//
//		std::vector<glm::vec2> clusterAaccBH(3, glm::vec2(0.0f));
//		std::vector<glm::vec2> clusterBaccBH(3, glm::vec2(0.0f));
//		visHelp::NodeNodeBH(smallCluster, clusterAaccBH, smallClusterAngled, clusterBaccBH);
//
//		std::vector<glm::vec2> clusterAaccFMM(3, glm::vec2(0.0f));
//		std::vector<glm::vec2> clusterBaccFMM(3, glm::vec2(0.0f));
//		visHelp::NodeNodeFMM(smallCluster, clusterAaccFMM, smallClusterAngled, clusterBaccFMM);
//
//		std::cout << "MSE of BH: " << visHelp::getMSE(clusterAaccBH, clusterAaccNaive) << std::endl;
//		std::cout << "MSE of FMM: " << visHelp::getMSE(clusterAaccFMM, clusterAaccNaive) << std::endl;
//	}
//	*/
//};