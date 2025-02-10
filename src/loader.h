#ifndef LOADER_H
#define LOADER_H

#include <unsupported/Eigen/SparseExtra>
//#include <Eigen/Sparse>
//#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

class Loader
{
public:
	static float* loadMNIST(unsigned int* dataAmount, unsigned int* dataDimension, const char* path, int maxAmount)
	{
        // this is the format of the data https://yann.lecun.com/exdb/mnist/
        FILE* file = fopen(path, "rb"); // open file at path in mode rb (read buffer)

        uint32_t magicNumber = getNextFourBytes(file);
        uint32_t amount = getNextFourBytes(file);
        uint32_t rows = getNextFourBytes(file);
        uint32_t cols = getNextFourBytes(file);

        int totalBytes = amount * rows * cols;
        int maxTotalBytes = maxAmount * rows * cols;

        unsigned char* dataChar = new unsigned char[std::min(totalBytes, maxTotalBytes)];
        fread(dataChar, 1, std::min(totalBytes, maxTotalBytes), file);

        *dataAmount = std::min((int)amount, maxAmount);
        *dataDimension = rows * cols;
        float* data = new float[std::min(totalBytes, maxTotalBytes)];

        for (int i = 0; i < std::min(totalBytes, maxTotalBytes); i++)
        {
            data[i] = (float)dataChar[i];
        }

        delete[] dataChar;
        fclose(file);
        return data;
	}
    
    static std::vector<uint8_t> loadLabels(std::string path)
    {
        std::ifstream file(path, std::ios::binary);

        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
        }

        return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    static Eigen::SparseMatrix<double> loadPmatrix(std::string path)
    {
        std::ifstream file(path);
        if (file.is_open())
        {
            Eigen::SparseMatrix<double> Pmatrix;
            Eigen::loadMarket(Pmatrix, path);
            std::cout << "Matrix loaded successfully!" << std::endl;
            return Pmatrix;
            //Pmatrix.coeff(i, j)
            //Pmatrix.rows()
            //Pmatrix.cols()
        }
        else
        {
            std::cerr << "Failed to open " + (std::string)path + " file!" << std::endl;
        }

        Eigen::SparseMatrix<double> test;//remove this
        return test;
    }

private:
    static uint32_t getNextFourBytes(FILE* file)
    {
        unsigned char buffer[4];
        fread(buffer, 1, 4, file);
        return (uint32_t)(buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3]);
    }

};
#endif

//call like this
//std::filesystem::path currentPath = std::filesystem::current_path();
//std::filesystem::path newPath = currentPath / "data\\t10k-images.idx3-ubyte";
// dataP = Loader::loadMNIST(&dataPAmount, &dataPDimension, newPath.string().c_str(), 50);