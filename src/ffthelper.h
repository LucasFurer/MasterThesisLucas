#ifndef FFTHELPER_H
#define FFTHELPER_H

#define REAL 0
#define IMAG 1

#include<fftw3/fftw3.h>

class FFTHelper
{
public:
	static void fft(fftw_complex* in, fftw_complex* out, int N)
	{
		// create dft plan
		fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		// execute plan
		fftw_execute(plan);
		// cleanup
		fftw_destroy_plan(plan);
		fftw_cleanup();
	}

	static void ifft(fftw_complex* in, fftw_complex* out, int N)
	{
		// create idft plan
		fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
		// execute plan
		fftw_execute(plan);
		// cleanup
		fftw_destroy_plan(plan);
		fftw_cleanup();
		// scale output to obtain the exact inverse
		for (int i = 0; i < N; i++)
		{
			out[i][REAL] /= N;
			out[i][IMAG] /= N;
		}
	}

	static void displatComplex(fftw_complex* y, int N)
	{
		for (int i = 0; i < N; i++)
		{
			if (y[i][IMAG] < 0)
			{
				std::cout << y[i][REAL] << " - " << abs(y[i][IMAG]) << "i" << std::endl;
			}
			else
			{
				std::cout << y[i][REAL] << " + " << y[i][IMAG] << "i" << std::endl;
			}
		}
	}

	static void displayReal(fftw_complex* y, int N)
	{
		for (int i = 0; i < N; i++)
		{
			std::cout << y[i][REAL] << std::endl;
		}
	}

private:
};



#endif

/*
//code for in main
const int N = 10;
fftw_complex x[N];
fftw_complex y[N];
for (int i = 0; i < N; i++)
{
	x[i][REAL] = i;
	x[i][IMAG] = N-i-1;
}
FFTHelper::fft(x, y, N);
FFTHelper::displatComplex(x, N);
FFTHelper::displatComplex(y, N);
FFTHelper::ifft(y, x, N);
FFTHelper::displatComplex(x, N);
*/