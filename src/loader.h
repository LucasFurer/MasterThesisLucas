#ifndef LOADER_H
#define LOADER_H

class Loader
{
public:
	static void loadMNIST(float* data, unsigned int* dataAmount, unsigned int* dataDimension, const char* path)
	{
        FILE* file = fopen(path, "rb"); // open file at path in mode rb (read buffer)

        uint32_t magicNumber = getNextFourBytes(file);
        uint32_t amount = getNextFourBytes(file);
        uint32_t rows = getNextFourBytes(file);
        uint32_t cols = getNextFourBytes(file);

        int totalBytes = amount * rows * cols;

        unsigned char* dataChar = new unsigned char[totalBytes];
        fread(dataChar, 1, totalBytes, file);

        *dataAmount = amount;
        *dataDimension = rows * cols;
        data = new float[totalBytes];
        for (int i = 0; i < totalBytes; i++)
        {
            data[i] = (float)dataChar[i];
        }

        fclose(file);
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
