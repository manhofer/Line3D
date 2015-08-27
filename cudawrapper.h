#ifndef I3D_LINE3D_CUDAWRAPPER_H_
#define I3D_LINE3D_CUDAWRAPPER_H_

/*
Line3D - Line-based Multi View Stereo
Copyright (C) 2015  Manuel Hofer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// external
#include "math_constants.h"
#include "cuda.h"
#include "helper_math.h"

// internal
#include "commons.h"
#include "sparsematrix.h"
#include "dataArray.h"

namespace L3D
{
    // constants CPU
    const unsigned int L3D_CU_BLOCK_SIZE_C = 16;

    // constants GPU
    __device__ const float L3D_EPS_G = 1e-12;
    __device__ const float L3D_COLLIN_AFF_T_G = 0.50f;
    __device__ const float L3D_MIN_OVERLAP_LOWER_T_G = 0.10f;
    __device__ const float L3D_MIN_OVERLAP_UPPER_T_G = 0.30f;

    // compute pairwise 2D line segment collinearity score
    extern void compute_collinearity(L3D::DataArray<float>* segments,
                                     L3D::DataArray<float>* relation,
                                     const float collin_s);

    // perform segment matching
    extern void compute_pairwise_matches(L3D::DataArray<float>* segments_src,
                                         L3D::DataArray<float>* RtKinv_src,
                                         L3D::DataArray<float4>* segments_tgt,
                                         L3D::DataArray<float>* RtKinv_tgt,
                                         L3D::DataArray<float>* camCenters_tgt,
                                         const float3 camCenter_src,
                                         L3D::DataArray<float>* fundamentals,
                                         L3D::DataArray<float>* projections,
                                         L3D::DataArray<int2>* offsets,
                                         std::list<unsigned int>& toBeMatched,
                                         std::list<L3D::L3DMatchingPair>& matches,
                                         std::map<unsigned int,unsigned int>& local2global,
                                         const unsigned int maxSegments, const unsigned int vID,
                                         const float uncertainty_k_upper,
                                         const float uncertainty_k_lower,
                                         const float sigma_p, const float sigma_a,
                                         const bool verify3D, const float spatial_k,
                                         float& median_depth,
                                         const bool verbose, const std::string prefix);

    // replicator dynamics diffusion [M.Donoser, BMVC'13]
    extern void replicator_dynamics_diffusion(L3D::SparseMatrix* &W, const bool verbose,
                                              const std::string prefix);
}

#endif //I3D_LINE3D_CUDAWRAPPER_H_
