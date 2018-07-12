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
// Hash and un-hash functions
#define P_hash 999
#define Q_hash 999
#define PQ_hash 998001
inline int hash(int m, int n, int k){
    return PQ_hash*m + Q_hash*n + k;
}
inline Triplet_mnk hash_back(int hash){
    int m = hash / PQ_hash;
    hash -= PQ_hash*m;
    int n = hash / Q_hash;
    hash -= Q_hash*n;
    int k = hash;
    return Triplet_mnk({m, n, k});
}

//===============================================================================
// Get block sizes defined in libcusmm
inline void get_blocksizes(std::vector<int>& v, std::unordered_map<int, Kernel_parameters > const& ht){
    for(auto it = ht.begin(); it != ht.end(); ++it){
        int h_mnk = it->first;
        Triplet_mnk v_mnk = hash_back(h_mnk);
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
