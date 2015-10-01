#ifndef I3D_LINE3D_COMMONS_H_
#define I3D_LINE3D_COMMONS_H_

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
#include "eigen3/Eigen/Eigen"
#include "cuda.h"
#include "helper_math.h"

// internal
#include "serialization.h"
#include "clustering.h"

/**
 * Line3D - Constants
 * ====================
 * Default parameters, etc.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3D
{
    // feature detection
    #define L3D_DEF_MAX_IMG_WIDTH 1920
    #define L3D_DEF_MIN_LINE_LENGTH_F 0.005f
    #define L3D_DEF_MAX_NUM_SEGMENTS 3000
    #define L3D_DEF_LOAD_AND_STORE_SEGMENTS true

    // collinearity
    #define L3D_DEF_COLLINEARITY_S 2.0f
    #define L3D_DEF_COLLINEARITY_FOR_CLUSTERING true

    // matching
    #define L3D_DEF_MATCHING_NEIGHBORS 10
    #define L3D_DEF_UNCERTAINTY_UPPER_T 5.0f
    #define L3D_DEF_UNCERTAINTY_LOWER_T 1.0f
    #define L3D_DEF_MIN_BASELINE_T 0.25f
    // same as for GPU!
    #define L3D_DEF_SIGMA_P 3.5f
    #define L3D_DEF_SIGMA_A 10.0f

    // replicator dynamics diffusion
    #define L3D_DEF_PERFORM_RDD false

    // clustering
    #define L3D_MIN_AFFINITY 0.25f

    #define L3D_EPS 1e-12

    // 3D segment
    struct L3DSegment3D
    {
        Eigen::Vector3d P1_;
        Eigen::Vector3d P2_;
        Eigen::Vector3d dir_;
        float depth_p1_;
        float depth_p2_;
        unsigned int camID_;
        unsigned int segID_;
    };

    // 2D segment (sortable)
    class L3DSegment2D
    {
    public:
        L3DSegment2D() : camID_(0), segID_(0){}
        L3DSegment2D(unsigned int camID,
                     unsigned int segID) :
            camID_(camID), segID_(segID){}
        inline unsigned int camID() const {return camID_;}
        inline unsigned int segID() const {return segID_;}

        inline bool operator== (const L3DSegment2D& rhs) const {return ((camID_ == rhs.camID_) && (segID_ == rhs.segID_));}
        inline bool operator< (const L3DSegment2D& rhs) const {
            return ((camID_ < rhs.camID_) || (camID_ == rhs.camID_ && segID_ < rhs.segID_));
        }
        inline bool operator!= (const L3DSegment2D& rhs) const {return !((*this) == rhs);}
    private:
        unsigned int camID_;
        unsigned int segID_;
    };

    // selected match
    class L3DCorrespondenceRRW
    {
    public:
        L3DCorrespondenceRRW(){
           valid_ = false;
           confidence_ = 0.0f;
           confidence_old_ = 0.0f;
           score_ = 0.0f;
        }
        L3DCorrespondenceRRW(unsigned int id, float confidence,
                             L3D::L3DSegment3D src_seg3D,
                             L3D::L3DSegment2D src,
                             L3D::L3DSegment2D tgt,
                             bool valid=true) :
            id_(id),confidence_(confidence),
            confidence_old_(0.0f),score_(0.0f),
            src_seg3D_(src_seg3D),
            src_(src),tgt_(tgt),
            valid_(valid){}
        ~L3DCorrespondenceRRW(){}

        // data
        unsigned int id(){return id_;}
        float confidence(){return confidence_;}
        float confidence_old(){return confidence_old_;}
        float score(){return score_;}
        L3D::L3DSegment3D src_seg3D(){return src_seg3D_;}
        L3D::L3DSegment2D src(){return src_;}
        L3D::L3DSegment2D tgt(){return tgt_;}
        bool valid(){return valid_;}

        // update confidence
        void updateConfidence(float c){
            confidence_old_ = confidence_;
            confidence_ = c;
        }

        void normalizeConfidence(float c){
            confidence_old_ = confidence_;
            confidence_ /= c;
        }

        void setScore(float s){
            score_ = s;
        }

        void invalidate(){valid_ = false;}
        void validate(){valid_ = true;}

    private:
        unsigned int id_;
        float confidence_;
        float confidence_old_;
        float score_;
        L3D::L3DSegment3D src_seg3D_;
        L3D::L3DSegment2D src_;
        L3D::L3DSegment2D tgt_;
        bool valid_;
    };

    static bool sortSelectedMatchesByConf(L3DCorrespondenceRRW* sm1,
                                          L3DCorrespondenceRRW* sm2)
    {
        return (sm1->confidence() > sm2->confidence());
    }

    // visual neighbor
    struct L3DVisualNeighbor
    {
        unsigned int camID_;
        float similarity_;
    };

    static bool sortVisualNeighbors(const L3DVisualNeighbor vn1,
                                    const L3DVisualNeighbor vn2)
    {
        return (vn1.similarity_ > vn2.similarity_);
    }

    struct L3DscorePlusID
    {
        float score_;
        unsigned int id_;
    };

    static bool sortScorePlusID(const L3DscorePlusID si1,
                                const L3DscorePlusID si2)
    {
        return (si1.score_ > si2.score_);
    }

    // sort detected segments by length
    static bool sortSegmentsByLength(const float2 s1, const float2 s2)
    {
        return (s1.y > s2.y);
    }

    // sortable 3D point along line
    struct SortablePointOnLine3D
    {
        Eigen::Vector3d P_;
        unsigned int segID_3D_;
        unsigned int camID_;
        float dist_;
    };

    static bool sortPointsOnLine3D(const SortablePointOnLine3D p1,
                                   const SortablePointOnLine3D p2)
    {
        return (p1.dist_ < p2.dist_);
    }

    // class for clustered 3D line
    class L3DFinalLine3D
    {
    public:
        L3DFinalLine3D(std::list<L3D::L3DSegment2D> segments2D,
                       std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> > segments3D)
        {
            segments3D_ = segments3D;
            segments2D_ = segments2D;
        }

        std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >* segments3D(){
            return &segments3D_;
        }

        std::list<L3D::L3DSegment2D>* segments2D(){
            return &segments2D_;
        }

    private:
        // 3D segments (along line)
        std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> > segments3D_;
        // 2D references
        std::list<L3D::L3DSegment2D> segments2D_;
    };
}

#endif //I3D_LINE3D_COMMONS_H_
