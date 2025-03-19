#pragma once

//template <typename T>
class SymmetricTensor
{
public:
	float* data;


	SymmetricTensor()
	{

	}

	~SymmetricTensor()
	{
		delete[] data;

	}

private:

};