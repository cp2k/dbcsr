/*****************************************************************************
*  CP2K: A general program to perform molecular dynamics simulations        *
*  Copyright (C) 2000 - 2018  CP2K developers group                         *
*****************************************************************************/

#ifndef PARAMETERS_HASH_H
#define PARAMETERS_HASH_H

#include <array>
#include <vector>
#include <unordered_map>

typedef std::array<int, 3> Triplet_mnk;
typedef std::array<int, 8> Kernel_parameters;

//===============================================================================
// Hash and reverse-hash functions
const int hash_limit = HASH_LIMIT;
inline int hash(int m, int n, int k){
    return (m << EXP_DOUBLE) | (n << EXP) | k;
}
inline Triplet_mnk hash_reverse(int hash){
    int m = hash >> EXP_DOUBLE;
    int n = (hash >> EXP) & HASH_LIMIT;
    int k = hash & HASH_LIMIT;
    return Triplet_mnk({m, n, k});
}

//===============================================================================
// Get block sizes defined in libcusmm
inline void get_blocksizes(std::vector<int>& v, std::unordered_map<int, Kernel_parameters > const& ht){
    for(auto it = ht.begin(); it != ht.end(); ++it){
        int h_mnk = it->first;
        Triplet_mnk v_mnk = hash_reverse(h_mnk);
        v.push_back(v_mnk[0]);
        v.push_back(v_mnk[1]);
        v.push_back(v_mnk[2]);
    }
}
inline void get_libcusmm_triplets(std::vector<int>& v, std::unordered_map<int, Kernel_parameters > const& ht){
    for(auto it = ht.begin(); it != ht.end(); ++it){
        v.push_back(it->first);
    }
}

#endif
//EOF
