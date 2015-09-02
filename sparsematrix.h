#ifndef I3D_LINE3D_SPARSEMATRIX_H_
#define I3D_LINE3D_SPARSEMATRIX_H_

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

// internal
#include "clustering.h"
#include "dataArray.h"

/**
 * Line3D - Sparsematrix
 * ====================
 * Sparse GPU matrix.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3D
{
// matching pair
struct L3DMatchingPair
{
    // src_image
    unsigned int segID1_;
    // tgt_image
    unsigned int camID2_;
    unsigned int segID2_;
    // depths
    float4 depths_;
    // confidence
    float confidence_;
    // defines if mp is still active
    bool active_;

    // serialization
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & boost::serialization::make_nvp("segID1_", segID1_);
        ar & boost::serialization::make_nvp("camID2_", camID2_);
        ar & boost::serialization::make_nvp("segID2_", segID2_);
        ar & boost::serialization::make_nvp("depths_x", depths_.x);
        ar & boost::serialization::make_nvp("depths_y", depths_.y);
        ar & boost::serialization::make_nvp("depths_z", depths_.z);
        ar & boost::serialization::make_nvp("depths_w", depths_.w);
        ar & boost::serialization::make_nvp("confidence_", confidence_);
        ar & boost::serialization::make_nvp("active_", active_);
    }
};

static bool sortMatchingPairs(const L3DMatchingPair mp1,
                              const L3DMatchingPair mp2)
{
    if(mp1.segID1_ < mp2.segID1_)
        return true;
    else if(mp1.segID1_ == mp2.segID1_ && mp1.camID2_ < mp2.camID2_)
        return true;
    else if(mp1.segID1_ == mp2.segID1_ && mp1.camID2_ == mp2.camID2_ && mp1.segID2_ < mp2.segID2_)
        return true;
    else
        return false;
}

static bool sortMatchingPairsByConf(const L3DMatchingPair mp1,
                                    const L3DMatchingPair mp2)
{
    return (mp1.confidence_ > mp2.confidence_);
}

// sort entries for sparse affinity matrix
static bool sortAffEntriesByCol(const float4 a1, const float4 a2)
{
    // affinity in z-Coordinate
    if(int(a1.y) < int(a2.y))
        return true;
    else if(int(a1.y) == int(a2.y) && int(a1.x) < int(a2.x))
        return true;
    else
        return false;
}

static bool sortAffEntriesByRow(const float4 a1, const float4 a2)
{
    // affinity in z-Coordinate
    if(int(a1.x) < int(a2.x))
        return true;
    else if(int(a1.x) == int(a2.x) && int(a1.y) < int(a2.y))
        return true;
    else
        return false;
}

// segment with confidence (to best match)
struct L3DSegmentInScope
{
    unsigned int camID_;
    unsigned int segID_;
    float weight_;
};

// best match plus scope
struct L3DBestMatch
{
    L3D::L3DMatchingPair ref_;
    std::list<L3D::L3DSegmentInScope> scope_;
};

    class SparseMatrix
    {
    public:
        SparseMatrix(std::list<float4>& entries, const unsigned int num_rows_cols,
                     const float normalization_factor=1.0f,
                     const bool sort_by_row=false, const bool already_sorted=false);
        SparseMatrix(std::list<L3D::CLEdge>& entries, const unsigned int num_rows_cols,
                     const float normalization_factor=1.0f,
                     const bool sort_by_row=false, const bool already_sorted=false);
        SparseMatrix(SparseMatrix* M, const bool change_sorting=false);
        ~SparseMatrix();

        // check element sorting
        bool isRowSorted(){
            return row_sorted_;
        }
        bool isColSorted(){
            return !row_sorted_;
        }

        // data access
        unsigned int num_rows_cols(){
            return num_rows_cols_;
        }
        unsigned int num_entries(){
            return num_entries_;
        }

        // CPU/GPU data
        L3D::DataArray<float4>* entries(){
            return entries_;
        }
        L3D::DataArray<int>* start_indices(){
            return start_indices_;
        }

        // download entries to CPU
        void download(){
            entries_->download();
        }

    private:
        // CPU/GPU data
        L3D::DataArray<float4>* entries_;
        L3D::DataArray<int>* start_indices_;

        bool row_sorted_;
        unsigned int num_rows_cols_;
        unsigned int num_entries_;
    };
}

#endif //I3D_LINE3D_SPARSEMATRIX_H_
