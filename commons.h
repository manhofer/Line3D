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
#include "Eigen/Eigen"
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
    #define L3D_DEF_SCALE_FACTOR 0.5f
    #define L3D_DEF_MIN_LINE_LENGTH_F 0.005f
    #define L3D_DEF_MAX_NUM_SEGMENTS 3000
    #define L3D_DEF_LOAD_AND_STORE_SEGMENTS true

    // collinearity
    #define L3D_DEF_COLLINEARITY_S 2.0f
    #define L3D_DEF_COLLINEARITY_FOR_CLUSTERING true

    // matching
    #define L3D_DEF_MATCHING_NEIGHBORS 12
    #define L3D_DEF_UNCERTAINTY_UPPER_T 4.5f
    #define L3D_DEF_UNCERTAINTY_LOWER_T 0.5f
    #define L3D_DEF_MIN_BASELINE_T 0.25f
    // same as for GPU!
    #define L3D_DEF_SIGMA_P 2.5f
    #define L3D_DEF_SIGMA_A 10.0f
    #define L3D_DEF_PERFORM_3D_VERIFICATION true

    // replicator dynamics diffusion
    #define L3D_DEF_PERFORM_RDD false
    #define L3D_RDD_MAX_ITER 10

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

    // sort entries for sparse affinity matrix (CLEdges)
    static bool sortCLEdgesByCol(const CLEdge a1, const CLEdge a2)
    {
        if(a1.j_ < a2.j_)
            return true;
        else if(a1.j_ == a2.j_ && a1.i_ < a2.i_)
            return true;
        else
            return false;
    }

    static bool sortCLEdgesByRow(const CLEdge a1, const CLEdge a2)
    {
        if(a1.i_ < a2.i_)
            return true;
        else if(a1.i_ == a2.i_ && a1.j_ < a2.j_)
            return true;
        else
            return false;
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
