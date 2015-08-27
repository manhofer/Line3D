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
#include "commons.h"
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
