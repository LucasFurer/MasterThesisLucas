#pragma once

#include <functional>
#include <utility>
#include <boost/sort/sort.hpp>
#include <cstdint>
#include <algorithm>


template <typename T>
struct MortonPoint
{
    uint32_t morton;
    T point;
};

template <typename T>
struct lessthan
{
    inline bool operator()(const MortonPoint<T>& x, const MortonPoint<T>& y) const
    {
        return x.morton < y.morton;
    }
};

template <typename T>
struct rightshift
{
    inline uint32_t operator()(const MortonPoint<T>& x, const unsigned offset) const
    {
        return x.morton >> offset;
    }
};