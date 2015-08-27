#ifndef I3D_LINE3D_VIEW_H_
#define I3D_LINE3D_VIEW_H_

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
#include "boost/filesystem.hpp"
#include "opencv/cv.h"

// internal
#include "segments.h"

/**
 * Line3D - View
 * ====================
 * Class that holds one reference image.
 * ====================
 * Author: M.Hofer, 2015
 */

namespace L3D
{
    class L3DView
    {
    public:
        L3DView(const unsigned int id, L3D::L3DSegments* segments,
                const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                const Eigen::Vector3d t,
                const unsigned int width, const unsigned int height,
                const float uncertainty_upper_px,
                const float uncertainty_lower_px,
                const std::string matchFilename,
                const std::string prefix);
        ~L3DView();

        // transform camera
        void transform(Eigen::Matrix4d& Qinv, double scale);

        // load existing pairwise matches
        void loadExistingMatches(std::list<L3D::L3DMatchingPair>& matches);
        void loadAndLocalizeExistingMatches(std::list<L3D::L3DMatchingPair>& matches,
                                            std::map<unsigned int,unsigned int>& global2local);
        void addMatches(std::list<L3D::L3DMatchingPair>& matches, bool remove_old=false,
                        bool only_best=false);

        // segment data access
        L3D::DataArray<float>* seg_coords();
        std::map<unsigned int,std::map<unsigned int,float> >* seg_collinearities();
        float4 getSegmentCoords(const unsigned int id);

        // camera data access
        Eigen::Matrix3d K(){return K_;}
        Eigen::Matrix3d Kinv(){return Kinv_;}
        Eigen::Matrix3d R(){return R_;}
        Eigen::Matrix3d Rt(){return Rt_;}
        Eigen::Matrix3d RtKinv(){return RtKinv_;}
        Eigen::Vector3d t(){return t_;}
        Eigen::Vector3d C(){return C_;}
        Eigen::MatrixXd P(){return P_;}
        Eigen::Vector3d principalPoint(){return principal_point_;}

        unsigned int width(){return width_;}
        unsigned int height(){return height_;}
        unsigned int id(){return id_;}

        // uncertainty estimator
        float uncertainty_k_upper(){return k_upper_;}
        float uncertainty_k_lower(){return k_lower_;}
        float median_depth(){return median_depth_;}

        void setMedianDepth(float value){
            median_depth_ = value;
        }

        // get uncertainty based on depth
        float get_lower_uncertainty(const float depth);
        float get_upper_uncertainty(const float depth);
        float get_uncertainty_sigma_squared(const float depth);

        // projective similarity
        float projective_similarity(const L3D::L3DSegment3D seg3D, const unsigned int seg2D_id,
                                    const float sigma);

        // unproject point with given depth
        L3D::L3DSegment3D unprojectSegment(const unsigned int id, const float depth_p1,
                                           const float depth_p2);

        // get the normalized ray through a 2D point and the camera center
        Eigen::Vector3d getNormalizedRay(const Eigen::Vector3d p);

        // draw lines into image
        void drawLines(cv::Mat& I, std::list<unsigned int> highlight);

        // baseline between views
        float baseline(L3D::L3DView* v);

        // specific spatial uncertainty slope (for scoring)
        float specificSpatialUncertaintyK(const float dist_px);

    private:
        // define spatial uncertainty
        void defineSpatialUncertainty();

        // camera data
        Eigen::Matrix3d K_;
        Eigen::Matrix3d Kinv_;
        Eigen::Matrix3d R_;
        Eigen::Matrix3d Rt_;
        Eigen::Matrix3d RtKinv_;
        Eigen::Vector3d t_;
        Eigen::Vector3d C_;
        Eigen::MatrixXd P_;
        Eigen::Vector3d principal_point_;

        float3 center_;

        // image data
        unsigned int id_;
        unsigned int width_;
        unsigned int height_;

        // matching data
        float uncertainty_upper_px_;
        float uncertainty_lower_px_;
        float k_upper_;
        float k_lower_;
        float median_depth_;

        // segment data
        L3D::L3DSegments* segments_;

        // system
        std::string raw_matches_file_;
        std::string final_matches_file_;
        std::string prefix_;
    };
}

#endif //I3D_LINE3D_VIEW_H_
