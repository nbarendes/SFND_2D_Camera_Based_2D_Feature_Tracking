/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
  
  
    vector<string> detector_type_name = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptor_type_name = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

  
/*********Counts_Keypoints file*************************/  
    ofstream detector_file;
  
    // create and open the .csv file
    detector_file.open("../TASK MP.7_Counts_Keypoints.csv");
/*******************************************************/    
  
/*********Counts_matched_Keypoints file*****************/    
    ofstream descriptor_file;  
    
    // create and open the .csv file
    descriptor_file.open("../TASK MP.8_Counts_matched_Keypoints.csv");  
  
/********************************************************/    
  
/******************Time_Keypoints file*******************/      
    ofstream det_desc_time_file;  

    // create and open the .csv file
    det_desc_time_file.open("../TASK MP.9_Time_Keypoints.csv");  
  
/********************************************************/   

    // write the file headers  
    detector_file <<"detector_type_name" << "," << "img-0" << "," << "img-1"<< "," << "img-2"  << "," << "img-3"  << "," << "img-4"  << "," << "img-5"  << "," << "img-6"  << "," << "img-7" << "," << "img-8"  << "," <<"img-9"  << std::endl;  
    descriptor_file <<"detector/descriptor_type_name"<< "," << "img-0 & img-1" << "," << "img-1 & img-2"<< "," << "img-2 & img-3"  << "," << "img-3 & img-4"  << "," << "img-4 & img-5"  << "," << "img-5 & img-6"  << "," << "img-6 & img-7"  << "," << "img-7 & img-8" << "," << "img-8 & img-9"  << std::endl; 
    det_desc_time_file <<"detector/descriptor"<< "," << "img-0 & img-1" << "," << "img-1 & img-2"<< "," << "img-2 & img-3"  << "," << "img-3 & img-4"  << "," << "img-4 & img-5"  << "," << "img-5 & img-6"  << "," << "img-6 & img-7"  << "," << "img-7 & img-8" << "," << "img-8 & img-9"  << std::endl; 
  
    // write data to the file        
    for(auto det : detector_type_name) // start loop detector_types
    {
        bool write_detector = false;

        for(auto des : descriptor_type_name) // start loop descriptor_types
        {
            if(det.compare("AKAZE")!=0 && des.compare("AKAZE")==0)
                continue;

            if(det.compare("AKAZE")==0 && des.compare("AKAZE")==0)
                continue;    

            dataBuffer.clear();
            
          
            // Write to detector keypoints number file
            if(!write_detector)
            {
                detector_file << det;
            }                
                     
            descriptor_file << det << "_" << des;
           
            det_desc_time_file << det << "_" << des;    


  
  
  

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        
         // Performance evaluation 3
         double t = (double)cv::getTickCount();
      
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
      

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        //dataBuffer.push_back(frame);
		if(dataBuffer.size()>= dataBufferSize)
        {
          dataBuffer.erase(dataBuffer.begin()); 
        }
        dataBuffer.push_back(frame);


        

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = det;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST")  == 0 || detectorType.compare("BRISK") == 0 || detectorType.compare("ORB")   == 0 || detectorType.compare("AKAZE") == 0 ||detectorType.compare("SIFT")  == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle
        vector<cv::KeyPoint>::iterator keypoint;
        vector<cv::KeyPoint> keypoints_roi;
      
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
           for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
                 {
                     if (vehicleRect.contains(keypoint->pt))
                     {  
                          cv::KeyPoint newKeyPoint;
                          newKeyPoint.pt = cv::Point2f(keypoint->pt);
                          newKeyPoint.size = 1;
                          keypoints_roi.push_back(newKeyPoint);
                     }
                  }

                    keypoints =  keypoints_roi;
                    cout << "IN ROI n= " << keypoints.size()<<" keypoints"<<endl;
                    
        }
      
      
        if(!write_detector)
        {
           detector_file  << ", " << keypoints.size();
        }  

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = des; // BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            string descriptorType;
            if (descriptorType.compare("SIFT") == 0) 
            {
               descriptorType == "DES_HOG";
            }
            else
            {
               descriptorType == "DES_BINARY";
            }              
          
          
            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
          
            //  Performance Evaluation 2
            descriptor_file << ", " << matches.size();
            
            //  Performance Evaluation 3
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            det_desc_time_file << ", " << 1000*t;
          
          
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
          
          
            if(!write_detector)
            {
                detector_file << endl;   
            }
            
            write_detector = true;

            descriptor_file << endl;
            det_desc_time_file << endl;
       
        }// eof loop over descriptor_types
    }// eof loop over detector_types

    detector_file.close();
    descriptor_file.close();
    det_desc_time_file.close();
  
  
    return 0;
}
