/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#ifndef PARAMETERS_UTILS_H
#define PARAMETERS_UTILS_H

#include <array>
#include <vector>
#include <unordered_map>
#include <functional>

typedef std::array<int, 3> Triplet;
typedef std::array<int, 8> KernelParameters;

namespace std
{
    template<> struct hash<Triplet>
    {
        size_t operator()(std::array<int,3> const& k) const noexcept
        {
            // the hash of an int is the int itself (perfect hash)
            size_t seed = k[0];
            // then mix the other hashes into it, see also boost::hash_combine
            seed ^= static_cast<size_t>(k[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= static_cast<size_t>(k[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

inline void get_libsmm_acc_triplets(std::vector<Triplet>& v, std::unordered_map<Triplet, KernelParameters> const& ht)
{
    for(auto it = ht.begin(); it != ht.end(); ++it)
        v.push_back(it->first);
}

#endif
