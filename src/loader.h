#ifndef LOADER_H
#define LOADER_H

class Loader
{
public:
	static float* loadMNIST(unsigned int* dataAmount, unsigned int* dataDimension, const char* path, int maxAmount)
	{
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

private:
    static uint32_t getNextFourBytes(FILE* file)
    {
        unsigned char buffer[4];
        fread(buffer, 1, 4, file);
        return (uint32_t)(buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3]);
    }

};
#endif
