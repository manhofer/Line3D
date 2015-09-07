#ifndef I3D_LINE3D_LINE3D_H_
#define I3D_LINE3D_LINE3D_H_

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
#include <map>

// external
#include "opencv/cv.h"
#include "eigen3/Eigen/Eigen"
#include "boost/filesystem.hpp"

// LSD
#include "lsd/lsd_opencv.hpp"

// internal
#include "commons.h"
#include "view.h"
#include "serialization.h"
#include "segments.h"
#include "cudawrapper.h"
#include "clustering.h"
#include "sparsematrix.h"
#include "dataArray.h"

/**
 * Line3D - Base Class
 * ====================
 * Line-based Multi-view Stereo
 * Reference: [add paper]
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3D
{
    class Line3D
    {
    public:
        Line3D(const std::string data_directory, const int matchingNeighbors=L3D_DEF_MATCHING_NEIGHBORS,
               const float uncertainty_t_upper_2D=L3D_DEF_UNCERTAINTY_UPPER_T,
               const float uncertainty_t_lower_2D=L3D_DEF_UNCERTAINTY_LOWER_T,
               const float sigma_p=L3D_DEF_SIGMA_P, const float sigma_a=L3D_DEF_SIGMA_A,
               const bool verify3D=L3D_DEF_PERFORM_3D_VERIFICATION, const float min_baseline=L3D_DEF_MIN_BASELINE_T,
               bool useCollinearity=L3D_DEF_COLLINEARITY_FOR_CLUSTERING, bool verbose=false);
        ~Line3D();

        // add a new image to the system
        void addImage(const unsigned int imageID, const cv::Mat image,
                      const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                      const Eigen::Vector3d t, std::list<unsigned int>& worldpointIDs,
                      const float scaleFactor=L3D_DEF_SCALE_FACTOR,
                      const bool loadAndStoreSegments=L3D_DEF_LOAD_AND_STORE_SEGMENTS);

        void addImage_fixed_sim(const unsigned int imageID, const cv::Mat image,
                                const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                                const Eigen::Vector3d t, std::map<unsigned int,float>& viewSimilarity,
                                const float scaleFactor=L3D_DEF_SCALE_FACTOR,
                                const bool loadAndStoreSegments=L3D_DEF_LOAD_AND_STORE_SEGMENTS);

        // reconstructs 3D model
        void compute3Dmodel(bool perform_diffusion=L3D_DEF_PERFORM_RDD);

        // get resulting 3D model
        void getResult(std::list<L3D::L3DFinalLine3D>& result);

        // get coordinates of a 2D segment (float4: p1x, p1y, p2x, p2y)
        float4 getSegment2D(L3D::L3DSegment2D& seg2D);

        // save model as STL file
        void save3DLinesAsSTL(std::list<L3D::L3DFinalLine3D>& result, std::string filename);

        // save model as txt file
        void save3DLinesAsTXT(std::list<L3D::L3DFinalLine3D>& result, std::string filename);

        // number of cameras
        unsigned int numCameras(){return views_.size();}

        // delete views etc.
        void reset();

    private:
        // system params
        bool verbose_;
        std::string prefix_;
        std::string separator_;
        std::string data_directory_;
        bool computation_;

        // view neighborhood information
        std::map<unsigned int,unsigned int> num_wps_;
        std::map<unsigned int,std::map<unsigned int,unsigned int> > common_wps_;
        std::map<unsigned int,std::map<unsigned int,float> > view_similarities_;
        std::map<unsigned int,std::map<unsigned int,bool> > worldpoints2views_;
        std::map<unsigned int,std::map<unsigned int,bool> > visual_neighbors_;
        std::map<unsigned int,std::map<unsigned int,Eigen::Matrix3d> > fundamentals_;

        // matching
        std::map<unsigned int,std::map<unsigned int,bool> > matched_;
        std::map<unsigned int,unsigned int> global2local_;
        std::map<unsigned int,unsigned int> local2global_;
        int matching_neighbors_;
        float min_baseline_;
        bool verify3D_;

        // scoring
        float uncertainty_upper_2D_;
        float uncertainty_lower_2D_;
        float sigma_p_;
        float sigma_a_;

        // final hypotheses
        std::map<L3D::L3DSegment2D,L3D::L3DCorrespondenceRRW> best_match_;
        std::map<L3D::L3DSegment2D,std::map<L3D::L3DSegment2D,bool> > potential_correspondences_;

        // clustering
        bool use_collinearity_;
        std::list<L3D::L3DFinalLine3D> clustered_result_;

        // geometry transformation
        Eigen::Matrix4d Qinv_;
        double transf_scale_;
        Eigen::Matrix3d transf_R_;
        Eigen::Vector3d transf_t_;
        // inverse transform
        double transf_scale_inv_;
        Eigen::Matrix3d transf_Rinv_;
        Eigen::Vector3d transf_tneg_;

        // LSD
        cv::Ptr<cv::LineSegmentDetector> ls_;

        // views
        std::map<unsigned int,L3D::L3DView*> views_;

        // detect line segments using the LSD algorithm
        bool detectLineSegments(const cv::Mat& image, std::list<float4> &lineSegments,
                                const unsigned int new_width, const unsigned int new_height,
                                const float min_length);

        // computes the length of a 2D line segment
        float segmentLength2D(const float4 coords);

        // adds worldpoint information to system
        void processWorldpointList(const unsigned int viewID, std::list<unsigned int>& wps);

        // for pmvs_data: set view similarity (according to overlap.txt)
        void setViewSimilarity(const unsigned int viewID, std::map<unsigned int,float>& sim);

        // find visually nearest neighbors among views
        void findVisualNeighbors();

        // transform geometry to avoid numerical imprecision
        void transformGeometry();
        Eigen::Vector3d inverseTransform(Eigen::Vector3d P);

        // computes a similarity transform between two pointsets
        void findSimilarityTransform(std::vector<Eigen::Vector3d>& input, Eigen::Vector3d& cog_in,
                                     std::vector<Eigen::Vector3d>& output, Eigen::Vector3d& cog_out);

        // computes an Euclidean transformation between two pointsets
        void euclideanTransformation(std::vector<Eigen::Vector3d>& input, Eigen::Vector3d& cog_in,
                                     std::vector<Eigen::Vector3d>& output, Eigen::Vector3d& cog_out);

        // applies found transformation to all cameras
        void applyTransformation();

        // match views with visual neighbors
        void matchViews();
        void performMatching(const unsigned int vID, std::list<L3D::L3DMatchingPair>& matches);

        // optimize correspondences
        void optimizeLocalMatches();
        void greedySelection();

        // cluster 2D segments to obtain final 3D model
        void clusterSegments2D(bool perform_diffusion);
        void performDiffusion(std::list<CLEdge>& A, const unsigned int num_rows_cols);
        void processClusteredSegments(L3D::CLUniverse* U, std::map<unsigned int,L3D::L3DSegment2D> &local2global);
        void untransformClusteredSegments(std::list<L3D::L3DSegment2D>& seg2D,
                                          std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& transformed3D);
        void alignClusteredSegments(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& transformed3D,
                                    std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >& seg3D,
                                    std::list<L3D::L3DSegment2D>& seg2D);
        void getLineEquation3D(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& seg3D,
                               L3D::L3DSegment3D& line3D);
        void projectToLine(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& unaligned,
                           std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >& aligned,
                           const L3D::L3DSegment3D line3D);


        // 2D similarity based on collinearity
        float similarity_coll3D(const L3D::L3DSegment3D seg1_3D, const L3D::L3DSegment3D seg2_3D);
        float distance_point2line_3D(const L3D::L3DSegment3D seg3D, const Eigen::Vector3d X);

        // compute fundamental matrices among visual neighbors
        void computeFundamentals(const unsigned int vID);
        Eigen::Matrix3d fundamental(const unsigned int view1,
                                    const unsigned int view2);
    };
}

#endif //I3D_LINE3D_LINE3D_H_
