#pragma once

#include <algorithm>
#include <vector>
#include <cmath>

//template <typename T>
class SymmetricTensor
{
public:
	float* data;
	int rank;
	int size;

	SymmetricTensor(int initRank, int initSize)
	{
		rank = initRank;
		size = initSize;

		int n = size + rank - 1;
		int k = rank;

		int symSize = binomialCoef(n, k);

		data = new float[symSize];
	}

	~SymmetricTensor()
	{
		delete[] data;
	}

	float get(std::vector<float>& indices)
	{
		std::sort(indices.begin(), indices.end());

		int flatIndex = 0;
		for (int n = 1; n <= rank; n++)
		{
			flatIndex += binomialCoef(indices[n - 1] + n - 1, n);
		}

		return data[flatIndex];
	}

	void set(std::vector<float>& indices, float value)
	{
		data[getFlatIndex(indices)] = value;
	}

	SymmetricTensor innerProduct(SymmetricTensor A, SymmetricTensor B)
	{
		int rankC = abs(A.rank - B.rank);

		return SymmetricTensor(rankC, A.size);
	}


private:
	inline int getFlatIndex(std::vector<float>& indices)
	{
		std::sort(indices.begin(), indices.end());

		int flatIndex = 0;
		for (int n = 1; n <= rank; n++)
		{
			flatIndex += binomialCoef(indices[n - 1] + n - 1, n);
		}

		return flatIndex;
	}

	int factorial(int n)
	{
		int result = 1;

		if (n == 0) { return result; }

		for (int i = 0; i < n; i++)
		{
			result *= n - i;
		}
	}

	int binomialCoef(int n, int k)
	{
		if (k < 0 || k > n) { return 0; }
		return factorial(n) / (factorial(k) * factorial(n - k));
	}
};