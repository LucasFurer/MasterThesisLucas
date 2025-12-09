#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>

class Timer
{
public:
	Timer() 
	{
		startTimer();
	}

	void startTimer()
	{
		start = std::chrono::high_resolution_clock::now();
	}

	void endTimer(std::string message)
	{
		end = std::chrono::high_resolution_clock::now();

		std::chrono::microseconds elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

		std::cout << message << ": " << formatWithDots(elapsed_time.count()) << " microseconds" << std::endl;
	}
private:
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::time_point();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::time_point();

	// helper to format number with dots every 3 digits
	std::string formatWithDots(long long value)
	{
		std::string s = std::to_string(value);
		int insertPosition = s.length() - 3;

		while (insertPosition > 0)
		{
			s.insert(insertPosition, ".");
			insertPosition -= 3;
		}

		return s;
	}
};
