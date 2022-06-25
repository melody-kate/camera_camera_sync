//
// Created by melody on 2022/6/16.
//

#ifndef SRC_CAMERA_CAMERA_SYNC_H
#define SRC_CAMERA_CAMERA_SYNC_H

#include <string>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <time.h>

class CameraCameraSync
{
public:
    void getImageTimeStamp(std::string oriDirName,std::string dstDirName);
    int getImageNumber();
    std::vector<std::pair<std::string,std::string>> imageTimeStampSyncFunction();
    double evaluateImageTimeStampSync(cv::Mat orgImage,cv::Mat dstImage);
    void spatialSynchronization(cv::Mat image1,cv::Mat image2);
private:
    std::vector<std::string> oriImageLists_;
    std::vector<std::string> dstImageLists_;
    float timeThreshold_;
    void getFiles(std::string path,std::vector<std::string>& files);
    double getBaseTime(std::string pngfilenames,std::string patt);
};



#endif //SRC_CAMERA_CAMERA_SYNC_H
