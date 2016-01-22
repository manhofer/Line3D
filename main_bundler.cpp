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

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>

// lib
#include "line3D.h"

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D");

    TCLAP::ValueArg<std::string> imgArg("i", "input_folder", "folder that contains the bundle.rd.out file", true, ".", "string");
    cmd.add(imgArg);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> image folder)", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching (-1 --> use all)", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);

    TCLAP::ValueArg<float> affLowerArg("a", "reprojection_error_lower_bound", "min uncertainty in image space for affinity estimation (t_l)", false, L3D_DEF_UNCERTAINTY_LOWER_T, "float");
    cmd.add(affLowerArg);

    TCLAP::ValueArg<float> affUpperArg("b", "reprojection_error_upper_bound", "max uncertainty in image space for affinity estimation (t_u)", false, L3D_DEF_UNCERTAINTY_UPPER_T, "float");
    cmd.add(affUpperArg);

    TCLAP::ValueArg<float> sigma_A_Arg("g", "sigma_a", "angle regularizer", false, L3D_DEF_SIGMA_A, "float");
    cmd.add(sigma_A_Arg);

    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer", false, L3D_DEF_SIGMA_P, "float");
    cmd.add(sigma_P_Arg);

    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);

    TCLAP::ValueArg<bool> verboseArg("v", "verbose", "more debug output is shown", false, false, "bool");
    cmd.add(verboseArg);

    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);

    TCLAP::ValueArg<bool> collinArg("e", "collinearity_flag", "try to cluster collinear segments", false, L3D_DEF_COLLINEARITY_FOR_CLUSTERING, "bool");
    cmd.add(collinArg);

    TCLAP::ValueArg<float> minBaselineArg("x", "min_image_baseline", "minimum baseline between matching images (world space)", false, L3D_DEF_MIN_BASELINE_T, "float");
    cmd.add(minBaselineArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string inputFolder = imgArg.getValue().c_str();
    std::string outputFolder = outputArg.getValue().c_str();
    if(outputFolder.length() == 0)
        outputFolder = inputFolder+"/Line3D/";

    int max_width = scaleArg.getValue();
    int neighbors = neighborArg.getValue();
    float max_uncertainty = fabs(affUpperArg.getValue());
    float min_uncertainty = fabs(affLowerArg.getValue());
    bool diffusion = diffusionArg.getValue();
    bool verbose = verboseArg.getValue();
    bool loadAndStore = loadArg.getValue();
    bool collinearity = collinArg.getValue();
    float sigma_a = fabs(sigma_A_Arg.getValue());
    float sigma_p = fabs(sigma_P_Arg.getValue());
    float min_baseline = fabs(minBaselineArg.getValue());

    std::string prefix = "[SYS] ";

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);
    std::string data_directory = outputFolder+"/L3D_data/";

    // create Line3D object
    L3D::Line3D* line3D = new L3D::Line3D(data_directory,neighbors,
                                          max_uncertainty,min_uncertainty,
                                          sigma_p,sigma_a,min_baseline,
                                          collinearity,verbose);

    // read bundle.rd.out
    std::ifstream bundle_file;
    bundle_file.open((inputFolder+"/bundle.rd.out").c_str());

    std::string bundle_line;
    std::getline(bundle_file,bundle_line); // ignore first line...
    std::getline(bundle_file,bundle_line);

    // read number of images and 3D points
    std::stringstream bundle_stream(bundle_line);
    unsigned int num_cams,num_points;
    bundle_stream >> num_cams >> num_points;

    std::cout << prefix << "num_cameras: " << num_cams << "  //  num_points: " << num_points << std::endl;

    if(num_cams == 0 || num_points == 0)
    {
        std::cerr << prefix << "No cameras and/or points in bundle file!" << std::endl;
        return -1;
    }

    // read camera data (sequentially)
    std::map<unsigned int,float> cams_focals;
    std::map<unsigned int,Eigen::Matrix3d> cams_rotation;
    std::map<unsigned int,Eigen::Vector3d> cams_translation;
    std::map<unsigned int,std::pair<float,float> > cams_distortion;
    for(unsigned int i=0; i<num_cams; ++i)
    {
        // focal_length,distortion
        double focal_length,dist1,dist2;
        std::getline(bundle_file,bundle_line);
        bundle_stream.str("");
        bundle_stream.clear();
        bundle_stream.str(bundle_line);
        bundle_stream >> focal_length >> dist1 >> dist2;

        cams_focals[i] = focal_length;
        cams_distortion[i] = std::pair<float,float>(dist1,dist2);

        // rotation
        Eigen::Matrix3d R;
        for(unsigned int j=0; j<3; ++j)
        {
            std::getline(bundle_file,bundle_line);
            bundle_stream.str("");
            bundle_stream.clear();
            bundle_stream.str(bundle_line);
            bundle_stream >> R(j,0) >> R(j,1) >> R(j,2);
        }

        // flip 2nd and 3rd line
        R(1,0) *= -1.0; R(1,1) *= -1.0; R(1,2) *= -1.0;
        R(2,0) *= -1.0; R(2,1) *= -1.0; R(2,2) *= -1.0;

        cams_rotation[i] = R;

        // translation
        std::getline(bundle_file,bundle_line);
        bundle_stream.str("");
        bundle_stream.clear();
        bundle_stream.str(bundle_line);
        Eigen::Vector3d t;
        bundle_stream >> t(0) >> t(1) >> t(2);

        // flip y and z!
        t(1) *= -1.0;
        t(2) *= -1.0;

        cams_translation[i] = t;
    }

    // read features (for image similarity calculation)
    std::map<unsigned int,std::list<unsigned int> > cams_worldpointIDs;
    for(unsigned int i=0; i<num_points; ++i)
    {
        // ignore first two...
        std::getline(bundle_file,bundle_line);
        std::getline(bundle_file,bundle_line);

        // view list
        std::getline(bundle_file,bundle_line);
        unsigned int num_views;

        std::istringstream iss(bundle_line);
        iss >> num_views;

        unsigned int camID,siftID;
        float posX,posY;
        for(unsigned int j=0; j<num_views; ++j)
        {
            iss >> camID >> siftID;
            iss >> posX >> posY;
            cams_worldpointIDs[camID].push_back(i);
        }
    }
    bundle_file.close();

    // load images sequentially
    for(unsigned int i=0; i<num_cams; ++i)
    {
        // transform ID
        std::stringstream id_str;
        id_str << std::setfill('0') << std::setw(8) << i;
        std::string fixedID = id_str.str();

        std::cout << prefix << "loading " << fixedID << " ..." << std::endl;

        // load image
        std::string img_filename = "";
        cv::Mat image;
        std::vector<boost::filesystem::wpath> possible_imgs;
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".jpg"));
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".JPG"));
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".png"));
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".PNG"));
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".jpeg"));
        possible_imgs.push_back(boost::filesystem::wpath(inputFolder+"/visualize/"+fixedID+".JPEG"));

        bool image_found = false;
        unsigned int pos = 0;
        while(!image_found && pos < possible_imgs.size())
        {
            if(boost::filesystem::exists(possible_imgs[pos]))
            {
                image_found = true;
                img_filename = possible_imgs[pos].string();
            }
            ++pos;
        }

        if(image_found)
        {
            // load image
            image = cv::imread(img_filename);

            // setup intrinsics
            float px = float(image.cols)/2.0f;
            float py = float(image.rows)/2.0f;
            float f = cams_focals[i];

            Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
            K(0,0) = f;
            K(1,1) = f;
            K(0,2) = px;
            K(1,2) = py;
            K(2,2) = 1.0;

            // undistort (if necessary)
            float d1 = cams_distortion[i].first;
            float d2 = cams_distortion[i].second;

            if(fabs(d1) > L3D_EPS || fabs(d2) > L3D_EPS)
            {
                std::cout << prefix << "undistorting... " << std::endl;

                cv::Mat I = cv::Mat_<double>::eye(3,3);
                cv::Mat cvK = cv::Mat_<double>::zeros(3,3);
                cvK.at<double>(0,0) = K(0,0);
                cvK.at<double>(1,1) = K(1,1);
                cvK.at<double>(0,2) = K(0,2);
                cvK.at<double>(1,2) = K(1,2);
                cvK.at<double>(2,2) = 1.0;

                cv::Mat cvDistCoeffs(4,1,CV_64FC1,cv::Scalar(0));
                cvDistCoeffs.at<double>(0) = d1;
                cvDistCoeffs.at<double>(1) = d2;
                cvDistCoeffs.at<double>(2) = 0.0;
                cvDistCoeffs.at<double>(3) = 0.0;

                cv::Mat undistort_map_x;
                cv::Mat undistort_map_y;

                cv::initUndistortRectifyMap(cvK,cvDistCoeffs,I,cvK,cv::Size(image.cols, image.rows),
                                            undistort_map_x.type(), undistort_map_x, undistort_map_y );
                cv::remap(image,image,undistort_map_x,undistort_map_y,cv::INTER_LINEAR,cv::BORDER_CONSTANT);
            }

            // add to system
            line3D->addImage(i,image,K,cams_rotation[i],cams_translation[i],cams_worldpointIDs[i],max_width,loadAndStore);
        }
        else
        {
            std::cerr << prefix << "warning: no image found! (only jpg and png supported)" << std::endl;
        }
    }

    // compute result
    line3D->compute3Dmodel(diffusion);

    // save end result
    std::list<L3D::L3DFinalLine3D> result;
    line3D->getResult(result);

    // set filename according to parameters
    std::stringstream str;
    str << "/line3D_result__";
    str << "W_" << max_width << "__";

    if(neighbors < 0)
        str << "N_ALL__";
    else
        str << "N_" << neighbors << "__";

    str << "tL_" << min_uncertainty << "__";
    str << "tU_" << max_uncertainty << "__";

    str << "sigmaP_" << sigma_p << "__";
    str << "sigmaA_" << sigma_a << "__";

    if(collinearity)
        str << "COLLIN__";
    else
        str << "NO_COLLIN__";

    if(diffusion)
        str << "DIFFUSION";
    else
        str << "NO_DIFFUSION";

    // save as STL
    line3D->save3DLinesAsSTL(result,outputFolder+str.str()+".stl");

    // save as txt
    line3D->save3DLinesAsTXT(result,outputFolder+str.str()+".txt");

    unsigned int num_indiv_segments = 0;
    std::list<L3D::L3DFinalLine3D>::iterator rit = result.begin();
    for(; rit!=result.end(); ++rit)
    {
        L3D::L3DFinalLine3D fl = *rit;
        num_indiv_segments += fl.segments3D()->size();
    }

    std::cout << prefix << "3D lines:        " << result.size() << std::endl;
    std::cout << prefix << "3D segments:     " << num_indiv_segments << std::endl;
    std::cout << prefix << "#images:         " << line3D->numCameras() << std::endl;

    // cleanup
    delete line3D;
}
