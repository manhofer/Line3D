#include "view.h"

namespace L3D
{
    //------------------------------------------------------------------------------
    L3DView::L3DView(const unsigned int id, L3D::L3DSegments* segments,
                     const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                     const Eigen::Vector3d t,
                     const unsigned int width, const unsigned int height,
                     const float uncertainty_upper_px,
                     const float uncertainty_lower_px,
                     const std::string matchFilename,
                     const std::string prefix)
    {
        // init
        K_ = K;
        R_ = R;
        t_ = t;

        principal_point_(0) = float(width)/2.0f;
        principal_point_(1) = float(height)/2.0f;
        principal_point_(2) = 1.0;

        Kinv_ = K_.inverse();
        Rt_ = R_.transpose();
        RtKinv_  = Rt_*Kinv_;
        C_ = Rt_ * (-1.0 * t_);
        center_ = make_float3(C_(0),C_(1),C_(2));

        // projection matrix
        Eigen::MatrixXd R_plus_t(3,4);
        R_plus_t.block<3,3>(0,0) = R_;
        R_plus_t.block<3,1>(0,3) = t_;
        P_ = K_ * R_plus_t;

        id_ = id;
        width_ = width;
        height_ = height;

        segments_ = segments;

        uncertainty_upper_px_ = uncertainty_upper_px;
        uncertainty_lower_px_ = uncertainty_lower_px;
        median_depth_ = 1.0f;

        raw_matches_file_ = matchFilename+"_raw.bin";
        final_matches_file_ = matchFilename+"_final.bin";
        prefix_ = prefix;

        // remove raw matches (if they exist)
        boost::filesystem::wpath file(raw_matches_file_);
        if(boost::filesystem::exists(file))
        {
            boost::filesystem::remove(file);
        }

        // remove final matches (if they exist)
        file = boost::filesystem::wpath(final_matches_file_);
        if(boost::filesystem::exists(file))
        {
            boost::filesystem::remove(file);
        }

        // compute linear spatial uncertainty parameters
        defineSpatialUncertainty();
    }

    //------------------------------------------------------------------------------
    L3DView::~L3DView()
    {
        if(segments_ != NULL)
            delete segments_;

        // remove raw matches
        boost::filesystem::wpath file(raw_matches_file_);
        if(boost::filesystem::exists(file))
        {
            boost::filesystem::remove(file);
        }

        // remove final matches
        file = boost::filesystem::wpath(final_matches_file_);
        if(boost::filesystem::exists(file))
        {
            boost::filesystem::remove(file);
        }
    }

    //------------------------------------------------------------------------------
    void L3DView::defineSpatialUncertainty()
    {
        // compute plane parallel to image plane
        Eigen::Vector3d n = RtKinv_*principal_point_;
        n.normalize();
        Eigen::Vector3d P = C_ + n; // depth = 1

        // shift principal point (x direction)
        Eigen::Vector3d pp_shifted1(principal_point_.x()+uncertainty_upper_px_,
                                    principal_point_.y(),1.0);
        Eigen::Vector3d pp_shifted2(principal_point_.x()+uncertainty_lower_px_,
                                    principal_point_.y(),1.0);

        // compute ray through shifted point
        Eigen::Vector3d d1 = RtKinv_*pp_shifted1;
        d1.normalize();
        Eigen::Vector3d d2 = RtKinv_*pp_shifted2;
        d2.normalize();

        // intersect with plane
        double t1 =  (P.dot(n) - n.dot(C_)) / (n.dot(d1));
        Eigen::Vector3d Q1 = C_ + t1 * d1;
        double t2 =  (P.dot(n) - n.dot(C_)) / (n.dot(d2));
        Eigen::Vector3d Q2 = C_ + t2 * d2;

        // compute distance
        double dist1 = (P-Q1).norm();
        double dist2 = (P-Q2).norm();

        k_upper_ = dist1;
        k_lower_ = dist2;
    }

    //------------------------------------------------------------------------------
    float L3DView::specificSpatialUncertaintyK(const float dist_px)
    {
        // compute plane parallel to image plane
        Eigen::Vector3d n = RtKinv_*principal_point_;
        n.normalize();
        Eigen::Vector3d P = C_ + n; // depth = 1

        // shift principal point (x direction)
        Eigen::Vector3d pp_shifted(principal_point_.x()+dist_px,
                                    principal_point_.y(),1.0);

        // compute ray through shifted point
        Eigen::Vector3d d = RtKinv_*pp_shifted;
        d.normalize();

        // intersect with plane
        double t =  (P.dot(n) - n.dot(C_)) / (n.dot(d));
        Eigen::Vector3d Q = C_ + t * d;

        // compute distance
        double dist = (P-Q).norm();

        return dist;
    }

    //------------------------------------------------------------------------------
    void L3DView::loadExistingMatches(std::list<L3D::L3DMatchingPair>& matches)
    {
        boost::filesystem::wpath file(raw_matches_file_);
        if(boost::filesystem::exists(file))
        {
            std::list<L3D::L3DMatchingPair> M;
            L3D::serializeFromFile(raw_matches_file_,M);
            matches.splice(matches.end(), M);
        }
    }

    //------------------------------------------------------------------------------
    void L3DView::addMatches(std::list<L3D::L3DMatchingPair>& matches, bool remove_old,
                             bool only_best)
    {
        if(only_best)
        {
            // only save the best match for each segment
            // (only makes sense when finalizing!)
            std::map<unsigned int,std::list<L3D::L3DMatchingPair> > best;
            std::list<L3D::L3DMatchingPair>::iterator it = matches.begin();
            for(; it!=matches.end(); ++it)
            {
                best[(*it).segID1_].push_back(*it);
            }

            matches.clear();
            std::map<unsigned int,std::list<L3D::L3DMatchingPair> >::iterator it2 = best.begin();
            for(; it2!=best.end(); ++it2)
            {
                it2->second.sort(L3D::sortMatchingPairsByConf);
                matches.push_back(it2->second.front());
            }
        }

        boost::filesystem::wpath file(raw_matches_file_);
        if(boost::filesystem::exists(file) && !remove_old)
        {
            std::list<L3D::L3DMatchingPair> M;
            L3D::serializeFromFile(raw_matches_file_,M);
            M.insert(M.end(),matches.begin(),matches.end());
            L3D::serializeToFile(raw_matches_file_,M);
        }
        else
        {
            L3D::serializeToFile(raw_matches_file_,matches);
        }
    }

    //------------------------------------------------------------------------------
    void L3DView::loadAndLocalizeExistingMatches(std::list<L3D::L3DMatchingPair>& matches,
                                                 std::map<unsigned int,unsigned int>& global2local)
    {
        boost::filesystem::wpath file(raw_matches_file_);
        if(boost::filesystem::exists(file))
        {
            std::list<L3D::L3DMatchingPair> M;
            L3D::serializeFromFile(raw_matches_file_,M);

            std::list<L3D::L3DMatchingPair>::iterator it = M.begin();
            for(; it!=M.end(); ++it)
            {
                L3D::L3DMatchingPair mp = *it;
                if(global2local.find(mp.camID2_) != global2local.end())
                {
                    mp.camID2_ = global2local[mp.camID2_];
                    matches.push_back(mp);
                }
                else
                {
                    std::cerr << prefix_ << "matches outside the local neighborhood in set!" << std::endl;
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    void L3DView::transform(Eigen::Matrix4d& Qinv, double scale)
    {
        // update translation
        t_ *= scale;

        // update orientation
        Eigen::MatrixXd Rt(3,4);
        Rt.block<3,3>(0,0) = R_;
        Rt.block<3,1>(0,3) = t_;

        Rt *= Qinv;

        R_ = Rt.block<3,3>(0,0);
        t_ = Rt.block<3,1>(0,3);

        // update derived values
        Rt_ = R_.transpose();
        P_ = K_ * Rt;

        RtKinv_ = Rt_*Kinv_;
        /*
        for(int r=0; r<3; ++r)
        {
            for(int c=0; c<3; ++c)
            {
                RtKinv_CPU_->data(c,r)[0] = RtKinv_(r,c);
            }
        }
        */
        C_ = Rt_ * (-1.0 * t_);
        center_ = make_float3(C_(0),C_(1),C_(2));

        // update spatial uncertainty
        defineSpatialUncertainty();
    }

    //------------------------------------------------------------------------------
    L3D::DataArray<float>* L3DView::seg_coords()
    {
        if(segments_ != NULL)
        {
            // segments available
            return segments_->segments();
        }
        return NULL;
    }

    //------------------------------------------------------------------------------
    float4 L3DView::getSegmentCoords(const unsigned int id)
    {
        if(segments_ != NULL && id < segments_->segments()->height())
        {
            return make_float4(segments_->segments()->dataCPU(0,id)[0],
                               segments_->segments()->dataCPU(1,id)[0],
                               segments_->segments()->dataCPU(2,id)[0],
                               segments_->segments()->dataCPU(3,id)[0]);
        }
        else
        {
            return make_float4(0,0,0,0);
        }
    }

    //------------------------------------------------------------------------------
    std::map<unsigned int,std::map<unsigned int,float> >* L3DView::seg_collinearities()
    {
        if(segments_ != NULL)
        {
            // segments available
            return segments_->collinearities();
        }
        return NULL;
    }

    //------------------------------------------------------------------------------
    L3D::L3DSegment3D L3DView::unprojectSegment(const unsigned int id, const float depth_p1,
                                                const float depth_p2)
    {
        L3D::L3DSegment3D seg3D;
        if(id > segments_->segments()->height())
        {
            std::cerr << prefix_ << "lineID out of range!" << std::endl;
            return seg3D;
        }

        // get 2D segment data
        Eigen::Vector3d p1(segments_->segments()->dataCPU(0,id)[0],
                           segments_->segments()->dataCPU(1,id)[0],
                           1.0);
        Eigen::Vector3d p2(segments_->segments()->dataCPU(2,id)[0],
                           segments_->segments()->dataCPU(3,id)[0],
                           1.0);

        // get rays
        Eigen::Vector3d ray1 = RtKinv_*p1;
        ray1.normalize();
        Eigen::Vector3d ray2 = RtKinv_*p2;
        ray2.normalize();

        seg3D.P1_ = C_ + ray1*depth_p1;
        seg3D.P2_ = C_ + ray2*depth_p2;

        // direction (normalized)
        seg3D.dir_ = seg3D.P2_-seg3D.P1_;
        seg3D.dir_.normalize();

        // cam ID
        seg3D.camID_ = id_;
        seg3D.segID_ = id;

        // depth
        seg3D.depth_p1_ = depth_p1;
        seg3D.depth_p2_ = depth_p2;

        return seg3D;
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d L3DView::getNormalizedRay(const Eigen::Vector3d p)
    {
        Eigen::Vector3d ray = RtKinv_*p;
        ray.normalize();
        return ray;
    }

    //------------------------------------------------------------------------------
    float L3DView::get_lower_uncertainty(const float depth)
    {
        if(depth < median_depth_)
            return k_lower_*depth;
        else
            return k_lower_*median_depth_;
    }

    //------------------------------------------------------------------------------
    float L3DView::get_upper_uncertainty(const float depth)
    {
        if(depth < median_depth_)
            return k_upper_*depth;
        else
            return k_upper_*median_depth_;
    }

    //------------------------------------------------------------------------------
    float L3DView::get_uncertainty_sigma_squared(const float depth)
    {
        float d1 = get_lower_uncertainty(depth);
        float d2 = get_upper_uncertainty(depth);

        return -(d2-d1)*(d2-d1)/(2.0f*logf(0.01f));
    }

    //------------------------------------------------------------------------------
    float L3DView::projective_similarity(const L3D::L3DSegment3D seg3D, const unsigned int seg2D_id,
                                         const float sigma)
    {
        if(seg2D_id >= segments_->segments()->height())
            return 0.0f;

        // project to image
        Eigen::Vector3d q1 = P_*Eigen::Vector4d(seg3D.P1_.x(),seg3D.P1_.y(),seg3D.P1_.z(),1.0);
        Eigen::Vector3d q2 = P_*Eigen::Vector4d(seg3D.P2_.x(),seg3D.P2_.y(),seg3D.P2_.z(),1.0);

        if(fabs(q1.z()) < L3D_EPS || fabs(q2.z()) < L3D_EPS)
            return 0.0f;

        q1 /= q1.z();
        q2 /= q2.z();

        // lines
        Eigen::Vector3d p1(segments_->segments()->dataCPU(0,seg2D_id)[0],
                           segments_->segments()->dataCPU(1,seg2D_id)[0],1.0);
        Eigen::Vector3d p2(segments_->segments()->dataCPU(2,seg2D_id)[0],
                           segments_->segments()->dataCPU(3,seg2D_id)[0],1.0);
        Eigen::Vector3d l1 = p1.cross(p2);
        l1.normalize();
        Eigen::Vector3d l2 = q1.cross(q2);
        l2.normalize();

        // compute distances
        float d1 = fabs((l1.x()*q1.x()+l1.y()*q1.y()+l1.z())/sqrtf(l1.x()*l1.x()+l1.y()*l1.y()));
        float d2 = fabs((l1.x()*q2.x()+l1.y()*q2.y()+l1.z())/sqrtf(l1.x()*l1.x()+l1.y()*l1.y()));
        float d3 = fabs((l2.x()*p1.x()+l2.y()*p1.y()+l2.z())/sqrtf(l2.x()*l2.x()+l2.y()*l2.y()));
        float d4 = fabs((l2.x()*p2.x()+l2.y()*p2.y()+l2.z())/sqrtf(l2.x()*l2.x()+l2.y()*l2.y()));

        float d = fmax(fmax(d1,d2),fmax(d3,d4));

        return expf(-d*d/(2.0f*sigma*sigma));
    }

    //------------------------------------------------------------------------------
    void L3DView::drawLines(cv::Mat& I, std::list<unsigned int> highlight)
    {
        I = cv::Mat::zeros(height_,width_,CV_8UC3);

        L3D::DataArray<float>* segs = segments_->segments();
        for(unsigned int i=0; i<segs->height(); ++i)
        {
            cv::Point p1(segs->dataCPU(0,i)[0],segs->dataCPU(1,i)[0]);
            cv::Point p2(segs->dataCPU(2,i)[0],segs->dataCPU(3,i)[0]);

            cv::line(I,p1,p2,cv::Scalar(255,255,255),4);
        }

        std::list<unsigned int>::iterator it = highlight.begin();
        for(; it!=highlight.end(); ++it)
        {
            unsigned int id = *it;
            if(id < segs->height())
            {
                cv::Point p1(segs->dataCPU(0,id)[0],segs->dataCPU(1,id)[0]);
                cv::Point p2(segs->dataCPU(2,id)[0],segs->dataCPU(3,id)[0]);

                cv::line(I,p1,p2,cv::Scalar(0,0,255),4);
            }
        }
    }

    //------------------------------------------------------------------------------
    float L3DView::baseline(L3D::L3DView* v)
    {
        return (C_ - v->C()).norm();
    }
}
