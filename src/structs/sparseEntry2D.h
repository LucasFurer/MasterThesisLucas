#pragma once

struct SparseEntryCSC2D
{
    int row;
    float value;

    SparseEntryCSC2D()
    {
        row = 0;
        value = 0.0f;
    }

    SparseEntryCSC2D(int initRow, float initValue)
    {
        row = initRow;
        value = initValue;
    }
};

struct SparseEntry2D
{
    int i;
    int j;
    float value;

    SparseEntry2D()
    {
        i = 0;
        j = 0;
        value = 0.0f;
    }

    SparseEntry2D(int initI, int initJ, float initValue)
    {
        i = initI;
        j = initJ;
        value = initValue;
    }
};
