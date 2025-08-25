#pragma once

struct SparseEntryCSC2D
{
    int row;
    float val;

    SparseEntryCSC2D()
    {
        row = 0;
        val = 0.0f;
    }

    SparseEntryCSC2D(int initRow, float initVal)
    {
        row = initRow;
        val = initVal;
    }
};

struct SparseEntryCOO2D
{
    int col;
    int row;
    float val;

    SparseEntryCOO2D()
    {
        col = 0;
        row = 0;
        val = 0.0f;
    }

    SparseEntryCOO2D(int initCol, int initRow, float initVal)
    {
        col = initCol;
        row = initRow;
        val = initVal;
    }
};
