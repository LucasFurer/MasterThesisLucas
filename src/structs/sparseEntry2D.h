#pragma once

struct SparseEntryCSC2D
{
    int row;
    double val;

    SparseEntryCSC2D()
    {
        row = 0;
        val = 0.0;
    }

    SparseEntryCSC2D(int initRow, double initVal)
    {
        row = initRow;
        val = initVal;
    }
};

struct SparseEntryCOO2D
{
    int col;
    int row;
    double val;

    SparseEntryCOO2D()
    {
        col = 0;
        row = 0;
        val = 0.0;
    }

    SparseEntryCOO2D(int initCol, int initRow, double initVal)
    {
        col = initCol;
        row = initRow;
        val = initVal;
    }
};
