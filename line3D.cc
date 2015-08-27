#include "line3D.h"

namespace L3D
{
    //------------------------------------------------------------------------------
    Line3D::Line3D(const std::string data_directory, const int matchingNeighbors,
                   const float uncertainty_t_upper_2D, const float uncertainty_t_lower_2D,
                   const float sigma_p, const float sigma_a,
                   const bool verify3D, const float min_baseline,
                   bool useCollinearity, bool verbose)
    {
        // init
        verbose_ = verbose;
        prefix_ = "[L3D] ";
        separator_ = "----------------------------------------";
        data_directory_ = data_directory;
        matching_neighbors_ = matchingNeighbors;
        uncertainty_upper_2D_ = fabs(uncertainty_t_upper_2D);
        uncertainty_lower_2D_ = fabs(uncertainty_t_lower_2D);
        computation_ = false;
        use_collinearity_ = useCollinearity;
        min_baseline_ = min_baseline;
        verify3D_ = verify3D;

        if(uncertainty_lower_2D_ < 1.0f)
            uncertainty_lower_2D_ = 1.0f;

        if(uncertainty_upper_2D_ <= uncertainty_lower_2D_)
            uncertainty_upper_2D_ = uncertainty_lower_2D_+1.0f;

        sigma_a_ = sigma_a;
        sigma_p_ = sigma_p;

        // create data directory
        boost::filesystem::path dir(data_directory_);
        boost::filesystem::create_directory(dir);

        // create LSD
        ls_ = cv::createLineSegmentDetectorPtr(cv::LSD_REFINE_ADV);

        // transform
        transf_scale_inv_ = 1.0;
        transf_Rinv_ = Eigen::Matrix3d::Identity();
        transf_tneg_ = Eigen::Vector3d(0.0,0.0,0.0);

        std::cout << prefix_ << "Line3D - http://www.icg.tugraz.at/ - AerialVisionGroup" << std::endl;
        std::cout << prefix_ << "(c) 2015, Manuel Hofer" << std::endl;
        std::cout << prefix_ << separator_ << std::endl;
    }

    //------------------------------------------------------------------------------
    Line3D::~Line3D()
    {
        std::cout << prefix_ << separator_ << std::endl;
        std::cout << prefix_ << ">>> EXITING SYSTEM <<<" << std::endl;
        std::cout << prefix_ << "cleaning up..." << std::endl;

        // cleanup
        reset();
    }

    //------------------------------------------------------------------------------
    void Line3D::reset()
    {
        // view neighborhood information
        num_wps_.clear();
        common_wps_.clear();
        view_similarities_.clear();
        worldpoints2views_.clear();
        visual_neighbors_.clear();
        fundamentals_.clear();

        // matching
        matched_.clear();
        global2local_.clear();
        local2global_.clear();

        // final hypotheses
        best_match_.clear();
        potential_correspondences_.clear();

        // result
        clustered_result_.clear();

        // cleanup
        std::map<unsigned int,L3D::L3DView*>::iterator it = views_.begin();
        for(; it!=views_.end(); ++it)
            delete it->second;

        views_.clear();

        computation_ = false;
    }

    //------------------------------------------------------------------------------
    void Line3D::addImage(const unsigned int imageID, const cv::Mat image,
                          const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                          const Eigen::Vector3d t, std::list<unsigned int>& worldpointIDs,
                          const float scaleFactor,
                          const bool loadAndStoreSegments)
    {
        if(computation_)
        {
            std::cerr << "reconstruction already performed! cannot add more images (try reset first)" << std::endl;
            return;
        }

        if(views_.size() == 0)
           std::cout << prefix_ << ">>> LOADING DATA <<<" << std::endl;

        std::cout << prefix_ << "adding image [" << imageID << "]" << std::endl;

        // check for unique ID
        if(views_.find(imageID) != views_.end())
        {
            std::cerr << prefix_ << "imageID already in use!" << std::endl;
            return;
        }
        else if(worldpointIDs.size() == 0)
        {
            std::cerr << prefix_ << "unlinked images cannot be added! (no worldpoints)" << std::endl;
            return;
        }

        // check image
        if(image.rows == 0 || image.cols == 0)
        {
            std::cerr << prefix_ << "image is empty!" << std::endl;
            return;
        }

        // compute new image sizes
        unsigned int new_width = round(float(image.cols)*scaleFactor);
        unsigned int new_height = round(float(image.rows)*scaleFactor);

        // check if features already computed
        std::stringstream str;
        if(use_collinearity_)
            str << "/segments_" << imageID << "_" << new_width << "x" << new_height << "_coll1.bin";
        else
            str << "/segments_" << imageID << "_" << new_width << "x" << new_height << "_coll0.bin";

        std::string feature_file = data_directory_+str.str();
        boost::filesystem::wpath file(feature_file);

        // remove if neccessary
        if(boost::filesystem::exists(file) && !loadAndStoreSegments)
        {
            boost::filesystem::remove(file);
        }

        L3D::L3DSegments* segments = NULL;
        if(boost::filesystem::exists(file) && loadAndStoreSegments)
        {
            if(verbose_)
                std::cout << prefix_ << "segments data found" << std::endl;

            // load segments
            segments = new L3D::L3DSegments();
            L3D::serializeFromFile(feature_file,*segments);
        }
        else
        {
            if(verbose_)
                std::cout << prefix_ << "performing line segment detection..." << std::endl;

            // detect line segments
            std::list<float4> lineSegments_vec;
            float min_length = L3D_DEF_MIN_LINE_LENGTH_F*sqrtf(float(image.rows*image.rows+image.cols*image.cols));
            if(detectLineSegments(image,lineSegments_vec,new_width,new_height,min_length))
            {
                // setup segment data
                segments = new L3D::L3DSegments(lineSegments_vec,use_collinearity_);

                // serialize to disk
                if(loadAndStoreSegments)
                    L3D::serializeToFile(feature_file,*segments);
            }
            else
            {
                // no segments detected
                return;
            }
        }

        if(verbose_)
            std::cout << prefix_ << "#segments: " << segments->num_segments() << " (final)" << std::endl;

        // create filenames for binarized matches
        std::stringstream str2;
        str2 << "/matches_" << imageID << "_" << new_width << "x" << new_height;
        std::string match_file = data_directory_+str2.str();

        // create view
        views_[imageID] = new L3D::L3DView(imageID,segments,K,R,t,
                                           image.cols,image.rows,
                                           uncertainty_upper_2D_,
                                           uncertainty_lower_2D_,
                                           match_file,
                                           prefix_);

        if(verbose_)
        {
            std::cout << prefix_ << "minimum uncertainty in depth=1: " << views_[imageID]->uncertainty_k_lower() << std::endl;
            std::cout << prefix_ << "maximum uncertainty in depth=1: " << views_[imageID]->uncertainty_k_upper() << std::endl;
        }

        // update neighborhood (worldpoint IDs)
        processWorldpointList(imageID,worldpointIDs);
    }

    //------------------------------------------------------------------------------
    void Line3D::addImage_fixed_sim(const unsigned int imageID, const cv::Mat image,
                                    const Eigen::Matrix3d K, const Eigen::Matrix3d R,
                                    const Eigen::Vector3d t, std::map<unsigned int,float>& viewSimilarity,
                                    const float scaleFactor,
                                    const bool loadAndStoreSegments)
    {
        if(computation_)
        {
            std::cerr << "reconstruction already performed! cannot add more images (try reset first)" << std::endl;
            return;
        }

        if(views_.size() == 0)
           std::cout << prefix_ << ">>> LOADING DATA <<<" << std::endl;

        std::cout << prefix_ << "adding image [" << imageID << "]" << std::endl;

        // check for unique ID
        if(views_.find(imageID) != views_.end())
        {
            std::cerr << prefix_ << "imageID already in use!" << std::endl;
            return;
        }
        else if(viewSimilarity.size() == 0)
        {
            std::cerr << prefix_ << "unlinked images cannot be added! (no view similarities)" << std::endl;
            return;
        }

        // check image
        if(image.rows == 0 || image.cols == 0)
        {
            std::cerr << prefix_ << "image is empty!" << std::endl;
            return;
        }

        // compute new image sizes
        unsigned int new_width = round(float(image.cols)*scaleFactor);
        unsigned int new_height = round(float(image.rows)*scaleFactor);

        // check if features already computed
        std::stringstream str;
        if(use_collinearity_)
            str << "/segments_" << imageID << "_" << new_width << "x" << new_height << "_coll1.bin";
        else
            str << "/segments_" << imageID << "_" << new_width << "x" << new_height << "_coll0.bin";

        std::string feature_file = data_directory_+str.str();
        boost::filesystem::wpath file(feature_file);

        // remove if neccessary
        if(boost::filesystem::exists(file) && !loadAndStoreSegments)
        {
            boost::filesystem::remove(file);
        }

        L3D::L3DSegments* segments = NULL;
        if(boost::filesystem::exists(file) && loadAndStoreSegments)
        {
            if(verbose_)
                std::cout << prefix_ << "segments data found" << std::endl;

            // load segments
            segments = new L3D::L3DSegments();
            L3D::serializeFromFile(feature_file,*segments);
        }
        else
        {
            if(verbose_)
                std::cout << prefix_ << "performing line segment detection..." << std::endl;

            // detect line segments
            std::list<float4> lineSegments_vec;
            float min_length = L3D_DEF_MIN_LINE_LENGTH_F*sqrtf(float(image.rows*image.rows+image.cols*image.cols));
            if(detectLineSegments(image,lineSegments_vec,new_width,new_height,min_length))
            {
                // setup segment data
                segments = new L3D::L3DSegments(lineSegments_vec,use_collinearity_);

                // serialize to disk
                if(loadAndStoreSegments)
                    L3D::serializeToFile(feature_file,*segments);
            }
            else
            {
                // no segments detected
                return;
            }
        }

        if(verbose_)
            std::cout << prefix_ << "#segments: " << segments->num_segments() << " (final)" << std::endl;

        // create filenames for binarized matches
        std::stringstream str2;
        str2 << "/matches_" << imageID << "_" << new_width << "x" << new_height;
        std::string match_file = data_directory_+str2.str();

        // create view
        views_[imageID] = new L3D::L3DView(imageID,segments,K,R,t,
                                           image.cols,image.rows,
                                           uncertainty_upper_2D_,
                                           uncertainty_lower_2D_,
                                           match_file,
                                           prefix_);

        if(verbose_)
        {
            std::cout << prefix_ << "minimum uncertainty in depth=1: " << views_[imageID]->uncertainty_k_lower() << std::endl;
            std::cout << prefix_ << "maximum uncertainty in depth=1: " << views_[imageID]->uncertainty_k_upper() << std::endl;
        }

        // update view similarity
        setViewSimilarity(imageID,viewSimilarity);
    }

    //------------------------------------------------------------------------------
    void Line3D::compute3Dmodel(bool perform_diffusion)
    {
        if(views_.size() < 4)
        {
            std::cerr << prefix_ << "not enough images! can't compute 3D model..." << std::endl;
            return;
        }

        computation_ = true;

        // reset everything that was computed previously
        matched_.clear();
        potential_correspondences_.clear();
        clustered_result_.clear();

        // find visual neighbors
        findVisualNeighbors();

        // transform geometry
        transformGeometry();

        // match views
        matchViews();

        // optimize correspondences (per cluster)
        optimizeLocalMatches();

        // cluster corresponding segments
        clusterSegments2D(perform_diffusion);
    }

    //------------------------------------------------------------------------------
    void Line3D::getResult(std::list<L3D::L3DFinalLine3D>& result)
    {
        result.clear();
        result = clustered_result_;
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsSTL(std::list<L3D::L3DFinalLine3D>& result, std::string filename)
    {
        std::ofstream file;
        file.open(filename.c_str());

        file << "solid lineModel" << std::endl;

        std::list<L3D::L3DFinalLine3D>::iterator it = result.begin();
        for(; it!=result.end(); ++it)
        {
            L3D::L3DFinalLine3D current = *it;

            std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >::iterator it2 = current.segments3D()->begin();
            for(; it2!=current.segments3D()->end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).first;
                Eigen::Vector3d P2 = (*it2).second;

                char x1[50];
                char y1[50];
                char z1[50];

                char x2[50];
                char y2[50];
                char z2[50];

                sprintf(x1,"%e",P1.x());
                sprintf(y1,"%e",P1.y());
                sprintf(z1,"%e",P1.z());

                sprintf(x2,"%e",P2.x());
                sprintf(y2,"%e",P2.y());
                sprintf(z2,"%e",P2.z());

                file << " facet normal 1.0e+000 0.0e+000 0.0e+000" << std::endl;
                file << "  outer loop" << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "   vertex " << x2 << " " << y2 << " " << z2 << std::endl;
                file << "   vertex " << x1 << " " << y1 << " " << z1 << std::endl;
                file << "  endloop" << std::endl;
                file << " endfacet" << std::endl;
            }
        }

        file << "endsolid lineModel" << std::endl;
        file.close();
    }

    //------------------------------------------------------------------------------
    void Line3D::save3DLinesAsTXT(std::list<L3D::L3DFinalLine3D>& result, std::string filename)
    {
        std::ofstream file;
        file.open(filename.c_str());

        std::list<L3D::L3DFinalLine3D>::iterator it = result.begin();
        for(; it!=result.end(); ++it)
        {
            L3D::L3DFinalLine3D current = *it;

            if(current.segments3D()->size() == 0)
                continue;

            // write 3D segments
            file << current.segments3D()->size() << " ";
            std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >::iterator it2 = current.segments3D()->begin();
            for(; it2!=current.segments3D()->end(); ++it2)
            {
                Eigen::Vector3d P1 = (*it2).first;
                Eigen::Vector3d P2 = (*it2).second;

                file << P1.x() << " " << P1.y() << " " << P1.z() << " ";
                file << P2.x() << " " << P2.y() << " " << P2.z() << " ";
            }

            // write 2D residuals
            file << current.segments2D()->size() << " ";
            std::list<L3D::L3DSegment2D>::iterator it3 = current.segments2D()->begin();
            for(; it3!=current.segments2D()->end(); ++it3)
            {
                file << (*it3).camID() << " " << (*it3).segID() << " ";
                float4 coords = getSegment2D(*it3);
                file << coords.x << " " << coords.y << " ";
                file << coords.z << " " << coords.w << " ";
            }

            file << std::endl;
        }

        file.close();
    }

    //------------------------------------------------------------------------------
    void Line3D::findVisualNeighbors()
    {
        visual_neighbors_.clear();
        std::cout << prefix_ << separator_ << std::endl;
        std::cout << prefix_ << ">>> FINDING VISUAL NEIGHBORS <<<" << std::endl;

        // compute similarites
        std::map<unsigned int,std::map<unsigned int,unsigned int> >::iterator it = common_wps_.begin();
        for(; it!=common_wps_.end(); ++it)
        {
            if(view_similarities_.find(it->first) != view_similarities_.end())
                continue;

            std::cout << prefix_ << "computing similarities for image [" << it->first << "]" << std::endl;
            std::map<unsigned int,unsigned int>::iterator n = it->second.begin();
            for(; n!=it->second.end(); ++n)
            {
                // compute similarity
                float sim = 2.0f*float(n->second)/(float(num_wps_[it->first]+num_wps_[n->first]));

                if(sim > L3D_EPS)
                {
                    view_similarities_[it->first][n->first] = sim;
                }
            }
        }

        // define visual neighbors
        std::map<unsigned int,std::map<unsigned int,float> >::iterator sit = view_similarities_.begin();
        for(; sit!=view_similarities_.end(); ++sit)
        {
            std::cout << prefix_ << "setting VNs for image [" << sit->first << "]" << std::endl;
            std::list<L3D::L3DVisualNeighbor> vn;
            std::map<unsigned int,float>::iterator n = sit->second.begin();
            for(; n!=sit->second.end(); ++n)
            {
                if(views_.find(n->first) != views_.end() && views_[sit->first]->baseline(views_[n->first]) > min_baseline_)
                {
                    // check existing VNs (baseline)
                    bool baseline_valid = true;
                    std::list<L3D::L3DVisualNeighbor>::iterator exVN = vn.begin();
                    for(; exVN!=vn.end() && baseline_valid; ++exVN)
                    {
                        L3D::L3DVisualNeighbor neigh = *exVN;
                        if(views_[neigh.camID_]->baseline(views_[n->first]) <= min_baseline_)
                            baseline_valid = false;
                    }

                    if(baseline_valid)
                    {
                        L3D::L3DVisualNeighbor neighbor;
                        neighbor.camID_ = n->first;
                        neighbor.similarity_ = n->second;
                        vn.push_back(neighbor);
                    }
                }
            }

            // sort by similarity
            vn.sort(L3D::sortVisualNeighbors);

            // limit number of neighbors
            if(matching_neighbors_ > 0 && int(vn.size()) > matching_neighbors_)
                vn.resize(matching_neighbors_);

            // store
            std::list<L3D::L3DVisualNeighbor>::iterator neigh_lst = vn.begin();
            for(; neigh_lst != vn.end(); ++neigh_lst)
                visual_neighbors_[sit->first][(*neigh_lst).camID_] = true;

            if(verbose_)
                std::cout << prefix_ << vn.size() << " visual neighbors found" << std::endl;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::transformGeometry()
    {
        std::cout << prefix_ << separator_ << std::endl;
        std::cout <<  prefix_ << ">>> TRANSFORMING SCENE GEOMETRY <<<" << std::endl;

        // reset fundamentals
        fundamentals_.clear();

        // mean point
        double size = views_.size();
        Eigen::Vector3d m(0.0,0.0,0.0);
        std::vector<Eigen::Vector3d> in_points;
        std::map<unsigned int,L3D::L3DView*>::iterator it = views_.begin();
        for(; it!=views_.end(); ++it)
        {
            m += it->second->C();
            in_points.push_back(it->second->C());
        }
        m /= size;

        // variance
        double q = 0.0;
        it = views_.begin();
        for(; it!=views_.end(); ++it)
        {
            q += (it->second->C() - m).norm();
        }
        q /= size;

        q = sqrtf(2.0)/q;

        // set matrix
        Eigen::MatrixXd T(4,4);
        T(0,0) = q;    T(0,1) = 0.0;  T(0,2) = 0.0;  T(0,3) = -q*m(0);
        T(1,0) = 0.0;  T(1,1) = q;    T(1,2) = 0.0;  T(1,3) = -q*m(1);
        T(2,0) = 0.0;  T(2,1) = 0.0;  T(2,2) = q;    T(2,3) = -q*m(2);
        T(3,0) = 0.0;  T(3,1) = 0.0;  T(3,2) = 0.0;  T(3,3) = 1.0;

        // transform points
        std::vector<Eigen::Vector3d> out_points;
        Eigen::Vector3d cog_out(0.0,0.0,0.0);
        for(unsigned int i=0; i<in_points.size(); ++i)
        {
            Eigen::Vector4d transf = T*Eigen::Vector4d(in_points[i](0),in_points[i](1),
                                                       in_points[i](2),1.0);

            Eigen::Vector3d t3(transf(0),transf(1),transf(2));
            cog_out += t3;
            out_points.push_back(t3);
        }
        cog_out /= size;

        if(verbose_)
        {
            std::cout <<  prefix_ << "COG_in   = [" << m.x() << " " << m.y() << " " << m.z() << "]" << std::endl;
            std::cout <<  prefix_ << "COG_out  = [" << cog_out.x() << " " << cog_out.y() << " " << cog_out.z() << "]" << std::endl;
            std::cout <<  prefix_ << "variance = " << q << std::endl;
        }

        // compute similarity transform
        std::cout <<  prefix_ << "computing similarity transform..." << std::endl;
        findSimilarityTransform(in_points,m,out_points,cog_out);

        // apply transformation
        applyTransformation();
    }

    //------------------------------------------------------------------------------
    void Line3D::matchViews()
    {
        std::cout << prefix_ << separator_ << std::endl;
        std::cout <<  prefix_ << ">>> MATCHING IMAGES <<<" << std::endl;

        // match images individually
        std::map<unsigned int,std::map<unsigned int,bool> >::iterator it = visual_neighbors_.begin();
        for(; it!=visual_neighbors_.end(); ++it)
        {
            std::cout << prefix_ << "matching image [" << it->first << "] with " << it->second.size() << " VNs" << std::endl;

            if(it->second.size() == 0)
                continue;

            // compute fundamental matrices
            computeFundamentals(it->first);

            // match with visual neighbors
            std::list<L3D::L3DMatchingPair> matches;
            performMatching(it->first,matches);

            if(verbose_)
            {
                size_t free_byte ;
                size_t total_byte ;
                cudaMemGetInfo( &free_byte, &total_byte);
                std::cout << prefix_ << "GPU_mem_free: " << free_byte/(1024*1024) << "MB - GPU_mem_total: " << total_byte/(1024*1024) << "MB" << std::endl;
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::performMatching(const unsigned int vID, std::list<L3D::L3DMatchingPair>& matches)
    {
        if(visual_neighbors_[vID].size() == 0)
        {
            std::cerr << prefix_ << "no visual neighbors for this image!" << std::endl;
            return;
        }

        // check if already matched with one or more neighbor(s)
        // and copy data to CPU/GPU matrices
        global2local_.clear();
        local2global_.clear();
        std::list<unsigned int> toBeMatched;
        unsigned int localID = 0;
        unsigned int maxFeatures = 0;
        unsigned int totalFeatures = 0;

        // CPU data
        L3D::DataArray<float>* fundamentals = new L3D::DataArray<float>(3,3*visual_neighbors_[vID].size());
        L3D::DataArray<float>* RtKinvs = new L3D::DataArray<float>(3,3*visual_neighbors_[vID].size());
        L3D::DataArray<float>* projections = new L3D::DataArray<float>(4,3*visual_neighbors_[vID].size());
        L3D::DataArray<int2>* offsets = new L3D::DataArray<int2>(visual_neighbors_[vID].size(),1);
        L3D::DataArray<float>* camCenters = new L3D::DataArray<float>(3,visual_neighbors_[vID].size());
        std::vector<float4> features_tgt_vec;

        std::map<unsigned int,bool>::iterator it = visual_neighbors_[vID].begin();
        for(; it!=visual_neighbors_[vID].end(); ++it)
        {
            // set local ID
            unsigned int locID = localID;
            global2local_[it->first] = locID;
            local2global_[locID] = it->first;
            ++localID;

            if(matched_[vID].find(it->first) == matched_[vID].end())
            {
                // not yet matched
                toBeMatched.push_back(locID);
            }

            // store fundamental matrix and Rt*Kinv
            Eigen::Matrix3d F = fundamentals_[vID][it->first];
            Eigen::Matrix3d RtKinv = views_[it->first]->RtKinv();
            for(int r=0; r<3; ++r)
            {
                for(int c=0; c<3; ++c)
                {
                    fundamentals->dataCPU(c,locID*3+r)[0] = F(r,c);
                    RtKinvs->dataCPU(c,locID*3+r)[0] = RtKinv(r,c);
                }
            }

            // store projection matrices
            Eigen::MatrixXd P = views_[it->first]->P();
            for(int r=0; r<3; ++r)
            {
                for(int c=0; c<4; ++c)
                {
                    projections->dataCPU(c,locID*3+r)[0] = P(r,c);
                }
            }

            // store camera center
            camCenters->dataCPU(0,locID)[0] = views_[it->first]->C().x();
            camCenters->dataCPU(1,locID)[0] = views_[it->first]->C().y();
            camCenters->dataCPU(2,locID)[0] = views_[it->first]->C().z();

            // store features
            unsigned int num_features = views_[it->first]->seg_coords()->height();
            if(num_features > maxFeatures)
                maxFeatures = num_features;

            for(unsigned int i=0; i<num_features; ++i)
            {
                features_tgt_vec.push_back(make_float4(views_[it->first]->seg_coords()->dataCPU(0,i)[0],
                                                       views_[it->first]->seg_coords()->dataCPU(1,i)[0],
                                                       views_[it->first]->seg_coords()->dataCPU(2,i)[0],
                                                       views_[it->first]->seg_coords()->dataCPU(3,i)[0]));
            }

            // set data offset
            offsets->dataCPU(locID,0)[0] = make_int2(totalFeatures,num_features);
            totalFeatures += num_features;
        }

        // move features to iu image
        L3D::DataArray<float4>* features_tgt = new L3D::DataArray<float4>(features_tgt_vec.size(),1,true,features_tgt_vec);

        // add source data
        L3D::DataArray<float>* RtKinv_src = new L3D::DataArray<float>(3,3);
        for(unsigned int r=0; r<3; ++r)
            for(unsigned int c=0; c<3; ++c)
                RtKinv_src->dataCPU(c,r)[0] = (views_[vID]->RtKinv())(r,c);

        // copy to GPU
        fundamentals->upload();
        projections->upload();
        RtKinvs->upload();
        camCenters->upload();
        features_tgt->upload();
        offsets->upload();
        RtKinv_src->upload();
        views_[vID]->seg_coords()->upload();
        float3 centerSrc = make_float3(views_[vID]->C().x(),
                                       views_[vID]->C().y(),
                                       views_[vID]->C().z());

        // load previous matches
        views_[vID]->loadAndLocalizeExistingMatches(matches,global2local_);
        if(verbose_)
            std::cout << prefix_ << "existing matches:  " << matches.size() << std::endl;

        // perform matching
        float median_depth = 1.0f;
        L3D::compute_pairwise_matches(views_[vID]->seg_coords(),RtKinv_src,features_tgt,
                                      RtKinvs,camCenters,centerSrc,
                                      fundamentals,projections,offsets,
                                      toBeMatched,matches,local2global_,
                                      maxFeatures,vID,
                                      views_[vID]->uncertainty_k_upper(),
                                      views_[vID]->uncertainty_k_lower(),
                                      sigma_p_,sigma_a_,verify3D_,
                                      views_[vID]->specificSpatialUncertaintyK(2.0f*sigma_p_),
                                      median_depth,
                                      verbose_,prefix_);

        // cleanup
        delete fundamentals;
        delete RtKinvs;
        delete projections;
        delete features_tgt;
        delete offsets;
        delete RtKinv_src;
        delete camCenters;
        views_[vID]->seg_coords()->removeFromGPU();

        // set median depth
        views_[vID]->setMedianDepth(median_depth);

        // store matches for other views
        std::map<unsigned int,std::list<L3D::L3DMatchingPair> > otherViews;
        std::list<L3D::L3DMatchingPair>::iterator mit = matches.begin();
        for(; mit!=matches.end(); ++mit)
        {
            L3D::L3DMatchingPair mp = *mit;
            unsigned int camID = mp.camID2_;
            if(visual_neighbors_[camID].find(vID) != visual_neighbors_[camID].end() &&
                    matched_[camID].find(vID) == matched_[camID].end())
            {
                L3D::L3DMatchingPair mp_rev;
                mp_rev.segID1_ = mp.segID2_;
                mp_rev.camID2_ = vID;
                mp_rev.segID2_ = mp.segID1_;
                mp_rev.confidence_ = 0.0f;
                mp_rev.depths_.x = mp.depths_.z;
                mp_rev.depths_.y = mp.depths_.w;
                mp_rev.depths_.z = mp.depths_.x;
                mp_rev.depths_.w = mp.depths_.y;
                mp_rev.active_ = true;
                otherViews[camID].push_back(mp_rev);
            }

            // store information about potential matches
            L3D::L3DSegment2D ref(vID,mp.segID1_);
            L3D::L3DSegment2D tgt(mp.camID2_,mp.segID2_);

            potential_correspondences_[ref][tgt] = true;
            potential_correspondences_[tgt][ref] = true;
        }

        std::map<unsigned int,std::list<L3D::L3DMatchingPair> >::iterator oit = otherViews.begin();
        for(; oit!=otherViews.end(); ++oit)
        {
            views_[oit->first]->addMatches(oit->second);
        }

        // set matched
        it = visual_neighbors_[vID].begin();
        for(; it!=visual_neighbors_[vID].end(); ++it)
        {
            matched_[vID][it->first] = true;
            if(visual_neighbors_[it->first].find(vID) != visual_neighbors_[it->first].end())
                matched_[it->first][vID] = true;
        }

        // store final matches
        views_[vID]->addMatches(matches,true,true);
    }

    //------------------------------------------------------------------------------
    void Line3D::optimizeLocalMatches()
    {
        std::cout << prefix_ << separator_ << std::endl;
        std::cout <<  prefix_ << ">>> OPTIMIZING CORRESPONDENCES (greedy) <<<" << std::endl;

        best_match_.clear();

        greedySelection();
    }

    //------------------------------------------------------------------------------
    void Line3D::greedySelection()
    {
        //std::list<L3D::L3DFinalLine3D> tmp;
        //std::list<L3D::L3DSegment2D> segments2D;

        // load correspondences for each image (store per segment)
        unsigned int clusterable = 0;
        unsigned int id = 0;
        std::map<unsigned int,L3D::L3DView*>::iterator it = views_.begin();
        unsigned int total_corrs = 0;
        for(; it!=views_.end(); ++it)
        {
            std::list<L3D::L3DMatchingPair> local_matches;
            it->second->loadExistingMatches(local_matches);
            total_corrs += local_matches.size();

            // store per segment
            std::map<L3DSegment2D,std::list<L3D::L3DMatchingPair> > matches;
            std::list<L3D::L3DMatchingPair>::iterator mit = local_matches.begin();
            for(; mit!=local_matches.end(); ++mit)
            {
                L3DSegment2D seg2D(it->first,(*mit).segID1_);
                matches[seg2D].push_back(*mit);
            }

            // sort by score
            std::map<L3DSegment2D,std::list<L3D::L3DMatchingPair> >::iterator it2 = matches.begin();
            for(; it2!=matches.end(); ++it2,++id)
            {
                L3DSegment2D src = it2->first;

                // sort by confidence
                it2->second.sort(L3D::sortMatchingPairsByConf);

                // define correspondence
                L3D::L3DMatchingPair mp = it2->second.front();

                // normalize confidence
                mp.confidence_ = fmin(mp.confidence_,1.0f);

                L3DSegment2D tgt(mp.camID2_,mp.segID2_);
                L3D::L3DSegment3D seg3D = views_[src.camID()]->unprojectSegment(src.segID(),mp.depths_.x,
                                                                                mp.depths_.y);
                L3D::L3DCorrespondenceRRW C(id,mp.confidence_,seg3D,src,tgt);
                C.setScore(mp.confidence_);

                // best match
                best_match_[src] = C;

                //std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> > segments3D;
                //segments3D.push_back(std::pair<Eigen::Vector3d,Eigen::Vector3d>(C.src_seg3D().P1_,C.src_seg3D().P2_));
                //L3D::L3DFinalLine3D tmp3D(segments2D,segments3D);
                //tmp.push_back(tmp3D);

                ++clusterable;
            }
        }

        if(verbose_)
        {
            std::cout << prefix_ << "#clusterable_segments:  " << clusterable << std::endl;
        }

        //save3DLinesAsSTL(tmp,data_directory_+"/unclustered.stl");
    }

    //------------------------------------------------------------------------------
    void Line3D::clusterSegments2D(bool perform_diffusion)
    {
        std::cout << prefix_ << separator_ << std::endl;
        std::cout <<  prefix_ << ">>> CLUSTERING 2D SEGMENTS (global) <<<" << std::endl;
        clustered_result_.clear();

        // create affinity matrix
        std::list<CLEdge> A;
        unsigned int localID = 0;
        std::map<L3D::L3DSegment2D,unsigned int> global2local;
        std::map<unsigned int,L3D::L3DSegment2D> local2global;

        std::map<L3D::L3DSegment2D,std::map<L3D::L3DSegment2D,bool> > used;

        std::cout << prefix_ << "computing affinity matrix..." << std::endl;

        std::map<L3D::L3DSegment2D,L3D::L3DCorrespondenceRRW>::iterator it = best_match_.begin();
        for(; it!=best_match_.end(); ++it)
        {
            L3D::L3DSegment2D src = it->first;
            std::list<CLEdge> localAffs;
            L3D::L3DSegment2D tgt;
            L3D::L3DCorrespondenceRRW C = it->second;

            if(!C.valid())
                continue;

            // affinities with segments from other views
            std::map<L3D::L3DSegment2D,bool>::iterator corrs = potential_correspondences_[src].begin();
            for(; corrs!=potential_correspondences_[src].end(); ++corrs)
            {
                tgt = corrs->first;

                if(used[src].find(tgt) != used[src].end())
                    continue;

                used[src][tgt] = true;
                used[tgt][src] = true;

                if(best_match_.find(tgt) != best_match_.end())
                {
                    // similarity
                    L3D::L3DCorrespondenceRRW C2 = best_match_[tgt];

                    if(C2.valid())
                    {
                        float w = 0.5f*(C.score()+C2.score())*similarity_coll3D(C.src_seg3D(),C2.src_seg3D());

                        if(w > L3D_MIN_AFFINITY)
                        {
                            // assign local ID
                            unsigned int locID;
                            if(global2local.find(src) == global2local.end())
                            {
                                // new ID
                                locID = localID;
                                ++localID;

                                global2local[src] = locID;
                                local2global[locID] = src;
                            }
                            else
                            {
                                // ID exists
                                locID = global2local[src];
                            }

                            // target ID
                            unsigned int tgtID;
                            if(global2local.find(tgt) == global2local.end())
                            {
                                // new ID
                                tgtID = localID;
                                ++localID;

                                global2local[tgt] = tgtID;
                                local2global[tgtID] = tgt;
                            }
                            else
                            {
                                // ID exists
                                tgtID = global2local[tgt];
                            }

                            // store
                            CLEdge e;
                            e.i_ = locID;
                            e.j_ = tgtID;
                            e.w_ = w;
                            localAffs.push_back(e);
                            e.j_ = locID;
                            e.i_ = tgtID;
                            localAffs.push_back(e);
                        }
                    }

                    // collinear segments with tgt
                    if(views_[tgt.camID()]->seg_collinearities()->find(tgt.segID()) != views_[tgt.camID()]->seg_collinearities()->end())
                    {
                        std::map<unsigned int,float>::iterator tgt_coll = views_[tgt.camID()]->seg_collinearities()->at(tgt.segID()).begin();
                        for(; tgt_coll!=views_[tgt.camID()]->seg_collinearities()->at(tgt.segID()).end(); ++tgt_coll)
                        {
                            L3D::L3DSegment2D tgtc(tgt.camID(),tgt_coll->first);

                            if(used[src].find(tgtc) != used[src].end())
                                continue;

                            used[src][tgtc] = true;
                            used[tgtc][src] = true;

                            if(best_match_.find(tgtc) != best_match_.end())
                            {
                                // similarity
                                L3D::L3DCorrespondenceRRW C3 = best_match_[tgtc];

                                if(C3.valid())
                                {
                                    float w = 0.5f*(C.score()+C3.score())*similarity_coll3D(C.src_seg3D(),C3.src_seg3D());

                                    if(w > 0.01f)
                                    {
                                        // assign local ID
                                        unsigned int locID;
                                        if(global2local.find(src) == global2local.end())
                                        {
                                            // new ID
                                            locID = localID;
                                            ++localID;

                                            global2local[src] = locID;
                                            local2global[locID] = src;
                                        }
                                        else
                                        {
                                            // ID exists
                                            locID = global2local[src];
                                        }

                                        // target ID
                                        unsigned int tgtcID;
                                        if(global2local.find(tgtc) == global2local.end())
                                        {
                                            // new ID
                                            tgtcID = localID;
                                            ++localID;

                                            global2local[tgtc] = tgtcID;
                                            local2global[tgtcID] = tgtc;
                                        }
                                        else
                                        {
                                            // ID exists
                                            tgtcID = global2local[tgtc];
                                        }

                                        // store
                                        CLEdge e;
                                        e.i_ = locID;
                                        e.j_ = tgtcID;
                                        e.w_ = w;
                                        localAffs.push_back(e);
                                        e.j_ = locID;
                                        e.i_ = tgtcID;
                                        localAffs.push_back(e);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // affinites with collinear segments
            if(views_[src.camID()]->seg_collinearities()->find(src.segID()) != views_[src.camID()]->seg_collinearities()->end())
            {
                std::map<unsigned int,float>::iterator c_it = views_[src.camID()]->seg_collinearities()->at(src.segID()).begin();
                for(; c_it!=views_[src.camID()]->seg_collinearities()->at(src.segID()).end(); ++c_it)
                {
                    unsigned int sID = c_it->first;
                    float collin_w = c_it->second;
                    tgt = L3D::L3DSegment2D(src.camID(),sID);

                    if(used[src].find(tgt) != used[src].end())
                        continue;

                    used[src][tgt] = true;
                    used[tgt][src] = true;

                    if(best_match_.find(tgt) != best_match_.end())
                    {
                        // similarity
                        L3D::L3DCorrespondenceRRW C2 = best_match_[tgt];

                        if(C2.valid())
                        {
                            float w = collin_w*0.5f*(C.score()+C2.score())*similarity_coll3D(C.src_seg3D(),C2.src_seg3D());

                            if(w > 0.01f)
                            {
                                // assign local ID
                                unsigned int locID;
                                if(global2local.find(src) == global2local.end())
                                {
                                    // new ID
                                    locID = localID;
                                    ++localID;

                                    global2local[src] = locID;
                                    local2global[locID] = src;
                                }
                                else
                                {
                                    // ID exists
                                    locID = global2local[src];
                                }

                                // target ID
                                unsigned int tgtID;
                                if(global2local.find(tgt) == global2local.end())
                                {
                                    // new ID
                                    tgtID = localID;
                                    ++localID;

                                    global2local[tgt] = tgtID;
                                    local2global[tgtID] = tgt;
                                }
                                else
                                {
                                    // ID exists
                                    tgtID = global2local[tgt];
                                }

                                // store
                                CLEdge e;
                                e.i_ = locID;
                                e.j_ = tgtID;
                                e.w_ = w;
                                localAffs.push_back(e);
                                e.j_ = locID;
                                e.i_ = tgtID;
                                localAffs.push_back(e);
                            }
                        }
                    }
                }
            }

            // copy affinites
            if(localAffs.size() > 0)
            {
                A.splice(A.end(),localAffs,localAffs.begin(),localAffs.end());
            }
        }

        global2local.clear();
        used.clear();

        if(verbose_)
        {
            std::cout << prefix_ << "A: #num_entries = " << A.size() << std::endl;
            std::cout << prefix_ << "A: #num_rows    = " << local2global.size() << std::endl;
        }

        if(A.size() == 0)
            return;

        if(perform_diffusion)
        {
            // diffusion
            std::cout << prefix_ << "replicator dynamics diffusion..." << std::endl;
            performDiffusion(A,local2global.size());
        }

        // perform clustering
        std::cout << prefix_ << "graph clustering..." << std::endl;

        CLUniverse* U = performClustering(A,local2global.size(),1.0f);

        processClusteredSegments(U,local2global);

        delete U;

        std::cout << prefix_ << clustered_result_.size() << " 3D lines found!" << std::endl;
    }

    //------------------------------------------------------------------------------
    void Line3D::performDiffusion(std::list<CLEdge>& A, const unsigned int num_rows_cols)
    {
        // create sparse GPU matrix
        L3D::SparseMatrix* W = new L3D::SparseMatrix(A,num_rows_cols);

        // perform RDD
        L3D::replicator_dynamics_diffusion(W,verbose_,prefix_);

        // update affinities (symmetrify)
        W->download();
        A.clear();

        std::map<int,std::map<int,float> > entries;
        for(unsigned int i=0; i<W->entries()->width(); ++i)
        {
            int s1 = W->entries()->dataCPU(i,0)[0].x;
            int s2 = W->entries()->dataCPU(i,0)[0].y;
            float w12 = W->entries()->dataCPU(i,0)[0].z;

            float w21 = w12;
            if(entries[s2].find(s1) != entries[s2].end())
            {
                // other one already processed
                w21 = entries[s2][s1];
            }

            float w = fmin(w12,w21);

            entries[s1][s2] = w;
            entries[s2][s1] = w;
        }

        std::map<int,std::map<int,float> >::iterator it = entries.begin();
        for(; it!=entries.end(); ++it)
        {
            std::map<int,float>::iterator it2 = it->second.begin();
            for(; it2!=it->second.end(); ++it2)
            {
                CLEdge e;
                e.i_ = it->first;
                e.j_ = it2->first;
                e.w_ = it2->second;
                A.push_back(e);
            }
        }

        // cleanup
        delete W;
    }

    //------------------------------------------------------------------------------
    void Line3D::processClusteredSegments(L3D::CLUniverse* U, std::map<unsigned int,L3D::L3DSegment2D> &local2global)
    {
        std::map<unsigned int,std::list<L3D::L3DSegment2D> > cluster2segments;
        std::map<unsigned int,std::map<unsigned int,bool> > cluster2cameras;

        std::map<unsigned int,L3D::L3DSegment2D>::iterator it = local2global.begin();
        for(; it!=local2global.end(); ++it)
        {
            unsigned int clID = U->find(it->first);
            L3D::L3DSegment2D seg = it->second;

            // store segment
            cluster2segments[clID].push_back(seg);
            // store camera
            cluster2cameras[clID][seg.camID()] = true;
        }

        if(verbose_)
            std::cout << prefix_ << "#clusters_total:  " << cluster2segments.size() << std::endl;

        //saveClustersToPly(cluster2segments,cluster2cameras,"clusters_raw",0,true);

        // estimate 3D lines for valid clusters (visible in >= 4 cameras)
        unsigned int valid_clusters = 0;
        //std::map<unsigned int,std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> > > tmp;
        std::map<unsigned int,std::list<L3D::L3DSegment2D> >::iterator cit = cluster2segments.begin();
        for(; cit!=cluster2segments.end(); ++cit)
        {
            if(cluster2cameras[cit->first].size() >= 4)
            {
                /*
                // DEBUG
                if(cit->first == 23977)
                {
                    visualizeCluster2D(cluster2segments[cit->first]);
                }
                */

                // get 3D data and transform back to original coordinate system
                std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> > transformed3D;
                untransformClusteredSegments(cit->second,transformed3D);

                // align segments
                std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> > segments3D;
                std::list<L3D::L3DSegment2D> segments2D;
                alignClusteredSegments(transformed3D,segments3D,segments2D);

                if(segments3D.size() > 0)
                {
                    // create clustered line
                    clustered_result_.push_back(L3DFinalLine3D(segments2D,segments3D));
                    ++valid_clusters;
                }

                // debug:
                //tmp[cit->first] = segments3D;
            }
        }
        //saveClustersToPly2(tmp,"clusters_final",0);

        if(verbose_)
            std::cout << prefix_ << "#clusters_valid:  " << valid_clusters << std::endl;
    }

    //------------------------------------------------------------------------------
    void Line3D::untransformClusteredSegments(std::list<L3D::L3DSegment2D>& seg2D,
                                              std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& transformed3D)
    {
        std::list<L3D::L3DSegment2D>::iterator it = seg2D.begin();
        for(; it!=seg2D.end(); ++it)
        {
            // get hypothesis
            if(best_match_.find(*it) != best_match_.end())
            {
                L3D::L3DCorrespondenceRRW C = best_match_[*it];

                // transform 3D points
                Eigen::Vector3d P1 = inverseTransform(C.src_seg3D().P1_);
                Eigen::Vector3d P2 = inverseTransform(C.src_seg3D().P2_);

                transformed3D[*it] = std::pair<Eigen::Vector3d,Eigen::Vector3d>(P1,P2);
            }
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::getLineEquation3D(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& seg3D,
                                   L3D::L3DSegment3D& line3D)
    {
        if(seg3D.size() == 0)
            return;

        // init
        Eigen::Vector3d P(0,0,0);

        int n = seg3D.size()*2;
        Eigen::MatrixXd g_points(3,n);

        std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >::iterator it = seg3D.begin();
        int j = 0;
        double sz = 0.0;
        for(; it!=seg3D.end(); ++it)
        {
            Eigen::Vector3d P1 = it->second.first;
            Eigen::Vector3d P2 = it->second.second;

            P += P1;
            P += P2;

            g_points(0,j) = P1.x();
            g_points(1,j) = P1.y();
            g_points(2,j) = P1.z();
            ++j;

            g_points(0,j) = P2.x();
            g_points(1,j) = P2.y();
            g_points(2,j) = P2.z();
            ++j;

            sz += 2.0;
        }

        // get line equation
        P /= sz;

        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n,n)-(1.0/(double)(n))*Eigen::MatrixXd::Constant(n,n,1.0);
        Eigen::MatrixXd Scat = g_points*C*g_points.transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Scat, Eigen::ComputeThinU);

        Eigen::MatrixXd U;
        Eigen::VectorXd S;

        U = svd.matrixU();
        S = svd.singularValues();

        int maxPos;
        S.maxCoeff(&maxPos);

        Eigen::Vector3d dir = Eigen::Vector3d(U(0, maxPos), U(1, maxPos), U(2, maxPos));
        dir.normalize();

        line3D.P1_ = P;
        line3D.P2_ = P+dir;
        line3D.dir_ = dir;
    }

    //------------------------------------------------------------------------------
    void Line3D::alignClusteredSegments(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& transformed3D,
                                        std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >& seg3D,
                                        std::list<L3D::L3DSegment2D>& seg2D)
    {
        seg3D.clear();
        seg2D.clear();
        if(transformed3D.size() == 0)
            return;

        // estimate 3D line using referenced segments
        L3D::L3DSegment3D seg_full;
        getLineEquation3D(transformed3D,seg_full);

        // compute individual segments
        projectToLine(transformed3D,seg3D,seg_full);

        // add 2D segments
        std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >::iterator it = transformed3D.begin();
        for(; it!=transformed3D.end(); ++it)
        {
            seg2D.push_back(it->first);
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::projectToLine(std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >& unaligned,
                               std::list<std::pair<Eigen::Vector3d,Eigen::Vector3d> >& aligned,
                               const L3D::L3DSegment3D line3D)
    {
        // compute sortable point set
        std::list<L3D::SortablePointOnLine3D> sortable;
        std::map<L3D::L3DSegment2D,std::pair<Eigen::Vector3d,Eigen::Vector3d> >::iterator it = unaligned.begin();
        unsigned int segID = 0;

        Eigen::Vector3d min_point(0.0,0.0,0.0);
        Eigen::Vector3d max_point(0.0,0.0,0.0);
        double min_length = 0.0;
        double max_length = 0.0;

        for(; it!=unaligned.end(); ++it,++segID)
        {
            L3D::SortablePointOnLine3D sp1,sp2;

            // project onto line
            Eigen::Vector3d P1 = it->second.first;
            Eigen::Vector3d P2 = it->second.second;
            Eigen::Vector3d proj1 = line3D.P1_+(line3D.dir_.dot(P1-line3D.P1_)/(line3D.dir_.norm()*line3D.dir_.norm()))*line3D.dir_;
            Eigen::Vector3d proj2 = line3D.P1_+(line3D.dir_.dot(P2-line3D.P1_)/(line3D.dir_.norm()*line3D.dir_.norm()))*line3D.dir_;
            sp1.P_ = P1;
            sp2.P_ = P2;

            // check for extremal points
            double loc1 = line3D.dir_.dot(line3D.P1_ - proj1);

            if(loc1 <= min_length)
            {
                min_length = loc1;
                min_point = proj1;
            }

            if(loc1 >= max_length)
            {
                max_length = loc1;
                max_point = proj1;
            }

            double loc2 = line3D.dir_.dot(line3D.P1_ - proj2);

            if(loc2 <= min_length)
            {
                min_length = loc2;
                min_point = proj2;
            }

            if(loc2 >= max_length)
            {
                max_length = loc2;
                max_point = proj2;
            }

            // camera and segment data
            sp1.segID_3D_ = segID;
            sp2.segID_3D_ = segID;
            sp1.camID_ = it->first.camID();
            sp2.camID_ = it->first.camID();

            sortable.push_back(sp1);
            sortable.push_back(sp2);
        }

        // set distance to extremal points
        std::list<L3D::SortablePointOnLine3D>::iterator sit = sortable.begin();
        for(; sit!=sortable.end(); ++sit)
        {
            (*sit).dist_ = ((*sit).P_-min_point).norm();
        }

        // sort by distance to extremal point
        sortable.sort(L3D::sortPointsOnLine3D);

        // define individual segments
        std::map<unsigned int,unsigned int> open;
        std::map<unsigned int,bool> open_lines;
        bool opened = false;
        Eigen::Vector3d current_start(0,0,0);
        sit = sortable.begin();
        for(; sit!=sortable.end(); ++sit)
        {
            L3D::SortablePointOnLine3D pt = *sit;

            if(open_lines.find(pt.segID_3D_) == open_lines.end())
            {
                // opening
                open_lines[pt.segID_3D_] = true;

                if(open.find(pt.camID_) == open.end())
                    open[pt.camID_] = 1;
                else
                    ++open[pt.camID_];
            }
            else
            {
                // closing
                open_lines.erase(pt.segID_3D_);

                --open[pt.camID_];

                if(open[pt.camID_] == 0)
                    open.erase(pt.camID_);
            }

            if(opened && open.size() < 3)
            {
                std::pair<Eigen::Vector3d,Eigen::Vector3d> l(current_start,pt.P_);
                aligned.push_back(l);
                opened = false;
            }
            else if(!opened && open.size() >= 3)
            {
                current_start = pt.P_;
                opened = true;
            }
        }
    }

    //------------------------------------------------------------------------------
    float Line3D::similarity_coll3D(const L3D::L3DSegment3D seg1_3D, const L3D::L3DSegment3D seg2_3D)
    {
        // distances
        float d1 = distance_point2line_3D(seg2_3D,seg1_3D.P1_);
        float d2 = distance_point2line_3D(seg2_3D,seg1_3D.P2_);
        float min_d1 = views_[seg1_3D.camID_]->get_lower_uncertainty(seg1_3D.depth_p1_);
        float min_d2 = views_[seg1_3D.camID_]->get_lower_uncertainty(seg1_3D.depth_p2_);
        float sigma_sqr_d1 = views_[seg1_3D.camID_]->get_uncertainty_sigma_squared(seg1_3D.depth_p1_);
        float sigma_sqr_d2 = views_[seg1_3D.camID_]->get_uncertainty_sigma_squared(seg1_3D.depth_p2_);

        float sim1,sim2;
        if(d1 < min_d1)
            sim1 = 1.0f;
        else
            sim1 = expf(-(d1-min_d1)*(d1-min_d1)/(2.0f*sigma_sqr_d1));

        if(d2 < min_d2)
            sim2 = 1.0f;
        else
            sim2 = expf(-(d2-min_d2)*(d2-min_d2)/(2.0f*sigma_sqr_d2));

        float w_d = 0.0f;
        float w_d12 = fmin(sim1,sim2);

        float d3 = distance_point2line_3D(seg1_3D,seg2_3D.P1_);
        float d4 = distance_point2line_3D(seg1_3D,seg2_3D.P2_);
        float min_d3 = views_[seg2_3D.camID_]->get_lower_uncertainty(seg2_3D.depth_p1_);
        float min_d4 = views_[seg2_3D.camID_]->get_lower_uncertainty(seg2_3D.depth_p2_);
        float sigma_sqr_d3 = views_[seg2_3D.camID_]->get_uncertainty_sigma_squared(seg2_3D.depth_p1_);
        float sigma_sqr_d4 = views_[seg2_3D.camID_]->get_uncertainty_sigma_squared(seg2_3D.depth_p2_);

        float sim3,sim4;
        if(d3 < min_d3)
            sim3 = 1.0f;
        else
            sim3 = expf(-(d3-min_d3)*(d3-min_d3)/(2.0f*sigma_sqr_d3));

        if(d4 < min_d4)
            sim4 = 1.0f;
        else
            sim4 = expf(-(d4-min_d4)*(d4-min_d4)/(2.0f*sigma_sqr_d4));


        float w_d34 = fmin(sim3,sim4);
        w_d = fmin(w_d12,w_d34);

        /*
        // spatial regularizer
        float d1 = distance_point2line_3D(seg2_3D,seg1_3D.P1_);
        float d2 = distance_point2line_3D(seg2_3D,seg1_3D.P2_);
        float max_d1 = views_[seg1_3D.camID_]->get_upper_uncertainty(seg1_3D.depth_p1_);
        float max_d2 = views_[seg1_3D.camID_]->get_upper_uncertainty(seg1_3D.depth_p2_);

        float d3 = distance_point2line_3D(seg1_3D,seg2_3D.P1_);
        float d4 = distance_point2line_3D(seg1_3D,seg2_3D.P2_);
        float max_d3 = views_[seg2_3D.camID_]->get_upper_uncertainty(seg2_3D.depth_p1_);
        float max_d4 = views_[seg2_3D.camID_]->get_upper_uncertainty(seg2_3D.depth_p2_);

        if(d1 > max_d1 || d2 > max_d2 || d3 > max_d3 || d4 > max_d4)
            return 0.0f;

        // projective position similarity
        float sim1 = views_[seg1_3D.camID_]->projective_similarity(seg2_3D,seg1_3D.segID_,uncertainty_lower_2D_);
        float sim2 = views_[seg2_3D.camID_]->projective_similarity(seg1_3D,seg2_3D.segID_,uncertainty_lower_2D_);
        float w_d = fmin(sim1,sim2);
        */

        // angle
        float angle = acos(fmax(fmin(seg1_3D.dir_.dot(seg2_3D.dir_),1.0),-1.0))/M_PI*180.0f;
        if(angle > 90.0f)
            angle = 180.0f-angle;

        float w_a = expf(-angle*angle/(2.0f*sigma_a_*sigma_a_));

        // fuse
        float sim = fmin(w_d,w_a);

        if(sim <= 0.01f)
            return 0.0f;
        else
            return sim;
    }

    //------------------------------------------------------------------------------
    float Line3D::distance_point2line_3D(const L3D::L3DSegment3D seg3D, const Eigen::Vector3d X)
    {
        Eigen::Vector3d P1 = seg3D.P1_;
        Eigen::Vector3d dir = seg3D.dir_;

        Eigen::Vector3d proj = P1 + (dir * ((X - P1).transpose()) * dir);
        return (proj-X).norm();
    }

    //------------------------------------------------------------------------------
    void Line3D::findSimilarityTransform(std::vector<Eigen::Vector3d>& input, Eigen::Vector3d& cog_in,
                                         std::vector<Eigen::Vector3d>& output, Eigen::Vector3d& cog_out)
    {
        // get scale
        size_t const n = input.size();
        std::vector<double> scales(n);
        double scales_sum = 0.0;
        for(size_t i = 0; i < n; ++i)
        {
            double d1 = (input[i]-cog_in).norm();
            double d2 = (output[i]-cog_out).norm();
            scales[i] = d2/d1;
            scales_sum += scales[i];
        }
        transf_scale_ = scales_sum/double(n);

        // scale input points
        cog_in *= transf_scale_;
        for(size_t i = 0; i < n; ++i)
            input[i] *= transf_scale_;

        // Euclidean transformation
        euclideanTransformation(input,cog_in,output,cog_out);
        transf_t_ /= transf_scale_;
    }

    //------------------------------------------------------------------------------
    void Line3D::euclideanTransformation(std::vector<Eigen::Vector3d>& input, Eigen::Vector3d& cog_in,
                                         std::vector<Eigen::Vector3d>& output, Eigen::Vector3d& cog_out)
    {
        // init
        size_t const n = input.size();
        for(size_t i = 0; i < n; ++i)
        {
            input[i] -= cog_in;
            output[i] -= cog_out;
        }

        Eigen::Matrix3d H;
        H.setZero();
        for(size_t i = 0; i < n; ++i)
        {
            // outer product
            Eigen::Matrix3d outer = output[i] * input[i].transpose() ;
            H += outer;
        }

        // solve
        Eigen::Matrix3d U, Vt;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Vt = svd.matrixV().transpose();
        U = svd.matrixU();

        transf_R_ = U * Vt;
        if(transf_R_.determinant() < 0)
        {
            // JacobiSVD returns smallest eigenvalue in last row
            Vt.row(2) *= -1;
            transf_R_ = U * Vt;
        }
        transf_t_ = cog_out - transf_R_*(cog_in);
    }

    //------------------------------------------------------------------------------
    void Line3D::applyTransformation()
    {
        // init
        Eigen::Matrix4d Q;
        Q << transf_R_.row(0), transf_t_.x() * transf_scale_,
             transf_R_.row(1), transf_t_.y() * transf_scale_,
             transf_R_.row(2), transf_t_.z() * transf_scale_,
             0.0, 0.0, 0.0, 1.0;
        Qinv_ = Q.inverse();

        // store for back projection
        transf_scale_inv_ = 1.0/transf_scale_;
        transf_Rinv_ = transf_R_.transpose();
        transf_tneg_ = -transf_t_;

        // transform views
        std::map<unsigned int,L3D::L3DView*>::iterator it = views_.begin();
        for(; it!=views_.end(); ++it)
        {
            it->second->transform(Qinv_,transf_scale_);
        }
    }

    //------------------------------------------------------------------------------
    Eigen::Vector3d Line3D::inverseTransform(Eigen::Vector3d P)
    {
        // transform back to original coordinate system
        return transf_Rinv_*(P*transf_scale_inv_ + transf_tneg_);
    }

    //------------------------------------------------------------------------------
    bool Line3D::detectLineSegments(const cv::Mat& image, std::list<float4> &lineSegments,
                                    const unsigned int new_width, const unsigned int new_height,
                                    const float min_length)
    {
        // scale image
        cv::Mat img_scaled;
        float upscale_factor;
        unsigned int original_width = image.cols;
        unsigned int original_height = image.rows;
        if(new_width != original_width || new_height != original_height)
        {
            cv::resize(image,img_scaled,cv::Size(new_width,new_height));
            float w_diff = float(img_scaled.cols)/float(image.cols);
            float h_diff = float(img_scaled.rows)/float(image.rows);
            upscale_factor = 1.0f/(0.5f*(w_diff+h_diff));
        }
        else
        {
            img_scaled = image.clone();
            upscale_factor = 1.0f;
        }

        // convert to grayscale
        cv::Mat imgGray;
        if(img_scaled.channels() > 1)
            cv::cvtColor(img_scaled,imgGray,CV_RGB2GRAY);
        else
            imgGray = img_scaled;

        // detect lines
        std::vector<cv::Vec4f> lines;
        std::vector<double> width, prec, nfa;
        ls_->detect(imgGray, lines, width, prec, nfa);

        if(lines.size() == 0)
            return false;

        // filter by size
        std::vector<float4> lines_filtered;
        std::list<float2> pos_and_length;
        for(unsigned int i=0; i<lines.size(); ++i)
        {
            cv::Vec4f pts = lines[i];
            if(nfa[i] >= 0.0)
            {
                float4 coords = make_float4(pts[0],pts[1],pts[2],pts[3]);
                coords *= upscale_factor;

                float len = segmentLength2D(coords);

                if(len > min_length)
                {
                    lines_filtered.push_back(coords);
                    pos_and_length.push_back(make_float2(lines_filtered.size()-1,len));
                }
            }
            else
            {
                // should not happen...
                if(verbose_)
                    std::cerr << prefix_ << "negative logNFA..." << std::endl;
            }
        }

        // sort by size
        pos_and_length.sort(L3D::sortSegmentsByLength);

        if(pos_and_length.size() > L3D_DEF_MAX_NUM_SEGMENTS)
            pos_and_length.resize(L3D_DEF_MAX_NUM_SEGMENTS);

        // store
        std::list<float2>::iterator fs = pos_and_length.begin();
        unsigned int i=0;
        for(; fs!=pos_and_length.end(); ++i,++fs)
        {
            unsigned int pos = (*fs).x;

            float4 coords = lines_filtered[pos];
            lineSegments.push_back(coords);
        }

        return true;
    }

    //------------------------------------------------------------------------------
    void Line3D::processWorldpointList(const unsigned int viewID, std::list<unsigned int>& wps)
    {
        if(verbose_)
            std::cout << prefix_ << "updating view neighborhoods using " << wps.size() << " wps" << std::endl;

        num_wps_[viewID] = 0;
        std::list<unsigned int>::iterator it = wps.begin();
        for(; it!=wps.end(); ++it)
        {
            // check if new valid worldpoint emerged
            if(worldpoints2views_[*it].size() == 2)
            {
                // new 3-view worldpoint
                std::map<unsigned int,bool>::iterator wit = worldpoints2views_[*it].begin();
                unsigned int v1 = wit->first;
                ++wit;
                unsigned int v2 = wit->first;

                if(common_wps_[v1].find(v2) == common_wps_[v1].end())
                {
                    // views have no common worldpoint yet
                    common_wps_[v1][v2] = 1;
                    common_wps_[v2][v1] = 1;
                }
                else
                {
                    ++common_wps_[v1][v2];
                    ++common_wps_[v2][v1];
                }

                // increment number of worldpoints
                ++num_wps_[v1];
                ++num_wps_[v2];
            }

            // update worldpoint information
            if(worldpoints2views_[*it].size() >= 2)
            {
                // worldpoint already existing
                std::map<unsigned int,bool>::iterator wit = worldpoints2views_[*it].begin();
                for(; wit!=worldpoints2views_[*it].end(); ++wit)
                {
                    if(common_wps_[wit->first].find(viewID) == common_wps_[wit->first].end())
                    {
                        // views have no common worldpoint yet
                        common_wps_[wit->first][viewID] = 1;
                        common_wps_[viewID][wit->first] = 1;
                    }
                    else
                    {
                        ++common_wps_[wit->first][viewID];
                        ++common_wps_[viewID][wit->first];
                    }
                }

                ++num_wps_[viewID];
            }

            // add to list
            worldpoints2views_[*it][viewID] = true;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::setViewSimilarity(const unsigned int viewID, std::map<unsigned int,float>& sim)
    {
        std::map<unsigned int,float>::iterator it = sim.begin();
        for(; it!=sim.end(); ++it)
        {
            if(it->second > 0.01f)
                view_similarities_[viewID][it->first] = it->second;
        }
    }

    //------------------------------------------------------------------------------
    void Line3D::computeFundamentals(const unsigned int vID)
    {
        std::map<unsigned int,bool>::iterator it = visual_neighbors_[vID].begin();
        for(; it!=visual_neighbors_[vID].end(); ++it)
        {
            unsigned int vID2 = it->first;
            if(fundamentals_[vID].find(vID2) == fundamentals_[vID].end())
            {
                // compute F
                Eigen::Matrix3d F = fundamental(vID,vID2);
                Eigen::Matrix3d Ft = F.transpose();

                fundamentals_[vID][vID2] = F;
                fundamentals_[vID2][vID] = Ft;
            }
        }
    }

    //------------------------------------------------------------------------------
    Eigen::Matrix3d Line3D::fundamental(const unsigned int view1,
                                        const unsigned int view2)
    {
        L3D::L3DView* v1 = views_[view1];
        L3D::L3DView* v2 = views_[view2];

        Eigen::Matrix3d K1 = v1->K();
        Eigen::Matrix3d R1 = v1->R();
        Eigen::Vector3d t1 = v1->t();

        Eigen::Matrix3d K2 = v2->K();
        Eigen::Matrix3d R2 = v2->R();
        Eigen::Vector3d t2 = v2->t();

        Eigen::Matrix3d R = R2 * R1.transpose();
        Eigen::Vector3d t = t2 - R * t1;

        Eigen::Matrix3d T(3,3);
        T(0,0) = 0.0;    T(0,1) = -t.z(); T(0,2) = t.y();
        T(1,0) = t.z();  T(1,1) = 0.0;    T(1,2) = -t.x();
        T(2,0) = -t.y(); T(2,1) = t.x();  T(2,2) = 0.0;

        Eigen::Matrix3d E = T * R;
        Eigen::Matrix3d F = K2.transpose().inverse() * E * K1.inverse();
        return F;
    }

    //------------------------------------------------------------------------------
    float Line3D::segmentLength2D(const float4 coords)
    {
        float x = coords.x-coords.z;
        float y = coords.y-coords.w;
        return sqrtf(x*x+y*y);
    }

    //------------------------------------------------------------------------------
    float4 Line3D::getSegment2D(L3D::L3DSegment2D& seg2D)
    {
        if(views_.find(seg2D.camID()) == views_.end())
        {
            std::cerr << prefix_ << "no view with ID " << seg2D.camID() << "!" << std::endl;
            return make_float4(0,0,0,0);
        }

        return views_[seg2D.camID()]->getSegmentCoords(seg2D.segID());
    }
}
