#ifndef I3D_LINE3D_SEGMENTS_H_
#define I3D_LINE3D_SEGMENTS_H_

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

// std
#include <vector>
#include <map>

// external
#include "boost/serialization/serialization.hpp"

// internal
#include "commons.h"
#include "cudawrapper.h"
#include "serialization.h"
#include "dataArray.h"

/**
 * Line3D - Segments
 * ====================
 * Class that holds all segments and
 * collinearity/intersection information
 * for one specific view.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3D
{
    // segment container
    class L3DSegments
    {
    public:
        // basic constructors
        L3DSegments(){
            segments_ = NULL;
        }
        ~L3DSegments(){
            delete segments_;
        }

        // data constructor
        L3DSegments(std::list<float4>& segments, const bool collin)
        {
            // store segments
            segments_ = new L3D::DataArray<float>(4,segments.size());
            std::list<float4>::iterator it = segments.begin();
            for(unsigned int i=0; it!=segments.end(); ++i,++it)
            {
                segments_->dataCPU(0,i)[0] = (*it).x;
                segments_->dataCPU(1,i)[0] = (*it).y;
                segments_->dataCPU(2,i)[0] = (*it).z;
                segments_->dataCPU(3,i)[0] = (*it).w;
            }

            if(collin)
            {
                // compute collinearity
                L3D::DataArray<float>* relation = new L3D::DataArray<float>(segments_->height(),segments_->height(),true);
                segments_->upload();
                L3D::compute_collinearity(segments_,relation,L3D_DEF_COLLINEARITY_S);
                segments_->removeFromGPU();

                // download
                relation->download();
                relation->removeFromGPU();

                for(unsigned int i=0; i<relation->width()-1; ++i)
                {
                    for(unsigned int j=i+1; j<relation->height(); ++j)
                    {
                        float data = relation->dataCPU(i,j)[0];

                        // collinearity
                        if(data > 0.0f)
                        {
                            segment2collinearities_[i][j] = data;
                            segment2collinearities_[j][i] = data;
                        }
                    }
                }

                delete relation;
            }
        }

        // data access
        unsigned int num_segments(){
            if(segments_ != NULL)
                return segments_->height();
            else
                return 0;
        }
        L3D::DataArray<float>* segments(){
            return segments_;
        }

        std::map<unsigned int,std::map<unsigned int,float> >* collinearities(){
            return &segment2collinearities_;
        }

    private:
        // segment data
        L3D::DataArray<float>* segments_;
        std::map<unsigned int,std::map<unsigned int,float> > segment2collinearities_;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & boost::serialization::make_nvp("segment2collinearities_", segment2collinearities_);
            ar & boost::serialization::make_nvp("segments_", segments_);
        }
    };
}

#endif //I3D_LINE3D_SEGMENTS_H_
