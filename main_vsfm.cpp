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

    TCLAP::ValueArg<std::string> imgArg("i", "input_folder", "folder that contains the images", true, ".", "string");
    cmd.add(imgArg);

    TCLAP::ValueArg<std::string> nvmArg("m", "nvm_file", "full path to the VisualSfM result file (.nvm)", true, ".", "string");
    cmd.add(nvmArg);

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
    std::string nvmFile = nvmArg.getValue().c_str();
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

    // check if NVM file exists
    boost::filesystem::path nvm(nvmFile);
    if(!boost::filesystem::exists(nvm))
    {
        std::cerr << "NVM file " << nvmFile << " does not exist!" << std::endl;
        return -1;
    }

    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);
    std::string data_directory = outputFolder+"/L3D_data/";

    // create Line3D object
    L3D::Line3D* line3D = new L3D::Line3D(data_directory,neighbors,
                                          max_uncertainty,min_uncertainty,
                                          sigma_p,sigma_a,min_baseline,
                                          collinearity,verbose);

    // read NVM file
    std::ifstream nvm_file;
    nvm_file.open(nvmFile.c_str());

    std::string nvm_line;
    std::getline(nvm_file,nvm_line); // ignore first line...
    std::getline(nvm_file,nvm_line); // ignore second line...

    // read number of images
    std::getline(nvm_file,nvm_line);
    std::stringstream nvm_stream(nvm_line);
    unsigned int num_cams;
    nvm_stream >> num_cams;

    if(num_cams == 0)
    {
        std::cerr << prefix << "No aligned cameras in NVM file!" << std::endl;
        return -1;
    }

    // read camera data (sequentially)
    std::vector<std::string> cams_imgFilenames(num_cams);
    std::vector<float> cams_focals(num_cams);
    std::vector<Eigen::Matrix3d> cams_rotation(num_cams);
    std::vector<Eigen::Vector3d> cams_translation(num_cams);
    std::vector<float> cams_distortion(num_cams);
    for(unsigned int i=0; i<num_cams; ++i)
    {
        std::getline(nvm_file,nvm_line);

        // image filename
        std::string filename;

        // focal_length,quaternion,center,distortion
        double focal_length,quat0,quat1,quat2,quat3;
        double Cx,Cy,Cz,dist;

        nvm_stream.str("");
        nvm_stream.clear();
        nvm_stream.str(nvm_line);
        nvm_stream >> filename >> focal_length >> quat3 >> quat0 >> quat1 >> quat2;
        nvm_stream >> Cx >> Cy >> Cz >> dist;

        cams_imgFilenames[i] = filename;
        cams_focals[i] = focal_length;
        cams_distortion[i] = dist;

        // rotation
        Eigen::Matrix3d R;
        R(0,0) = 1.0-2.0*quat1*quat1-2.0*quat2*quat2;
        R(0,1) = 2.0*quat0*quat1-2.0*quat2*quat3;
        R(0,2) = 2.0*quat0*quat2+2.0*quat1*quat3;

        R(1,0) = 2.0*quat0*quat1+2.0*quat2*quat3;
        R(1,1) = 1.0-2.0*quat0*quat0-2.0*quat2*quat2;
        R(1,2) = 2.0*quat1*quat2-2.0*quat0*quat3;

        R(2,0) = 2.0*quat0*quat2-2.0*quat1*quat3;
        R(2,1) = 2.0*quat1*quat2+2.0*quat0*quat3;
        R(2,2) = 1.0-2.0*quat0*quat0-2.0*quat1*quat1;

        // translation
        Eigen::Vector3d C(Cx,Cy,Cz);
        Eigen::Vector3d t = -R*C;

        cams_translation[i] = t;
        cams_rotation[i] = R;
    }

    // read number of images
    std::getline(nvm_file,nvm_line); // ignore line...
    std::getline(nvm_file,nvm_line);
    nvm_stream.str("");
    nvm_stream.clear();
    nvm_stream.str(nvm_line);
    unsigned int num_points;
    nvm_stream >> num_points;

    // read features (for image similarity calculation)
    std::vector<std::list<unsigned int> > cams_worldpointIDs(num_cams);
    for(unsigned int i=0; i<num_points; ++i)
    {
        // 3D position
        std::getline(nvm_file,nvm_line);
        std::istringstream iss_point3D(nvm_line);
        double px,py,pz,colR,colG,colB;
        iss_point3D >> px >> py >> pz;
        iss_point3D >> colR >> colG >> colB;

        // measurements
        unsigned int num_views;
        iss_point3D >> num_views;

        unsigned int camID,siftID;
        float posX,posY;
        for(unsigned int j=0; j<num_views; ++j)
        {
            iss_point3D >> camID >> siftID;
            iss_point3D >> posX >> posY;
            cams_worldpointIDs[camID].push_back(i);
        }
    }
    nvm_file.close();

    // load images sequentially
    for(unsigned int i=0; i<num_cams; ++i)
    {
        // load image
        cv::Mat image = cv::imread(inputFolder+"/"+cams_imgFilenames[i]);

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
        float d = cams_distortion[i];

        if(fabs(d) > L3D_EPS)
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
            cvDistCoeffs.at<double>(0) = -d;
            cvDistCoeffs.at<double>(1) = 0.0;
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
