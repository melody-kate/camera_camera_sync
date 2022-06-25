//
// Created by melody on 2022/6/16.
//

#include "camera_camera_sync.h"
#include <dirent.h>
#include <opencv2/features2d/features2d.hpp>
#include <cstdlib>

void CameraCameraSync::getFiles(std::string path, std::vector<std::string> &files)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    char *basePath=const_cast<char *>(path.c_str());
    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);

    }

    while ((ptr=readdir(dir)) != NULL)
    {
        // current dir
        if(strcmp(ptr->d_name, ".")==0 || strcmp(ptr->d_name, "..")==0)
            continue;
        else if(ptr->d_type == 8) // file
            sprintf(base, "%s/%s", basePath, ptr->d_name);
        //puts(base);
        files.push_back(std::string(base));
    }
}

void CameraCameraSync::getImageTimeStamp(std::string oriDirName, std::string dstDirName)
{
    getFiles(oriDirName,oriImageLists_);
    getFiles(dstDirName,dstImageLists_);
    if(oriImageLists_.size()!=dstImageLists_.size())
    {
        std::cout<<"the two image lists not equal"<<std::endl;
        return;
    }
}

int CameraCameraSync::getImageNumber()
{
    if(oriImageLists_.size()!=dstImageLists_.size())
    {
        std::cout<<"the two image lists not equal"<<std::endl;
        return -1;
    }
    return oriImageLists_.size();
}

double CameraCameraSync::getBaseTime(std::string pngfilenames, std::string patt)
{
    size_t pattern=pngfilenames.find(patt);
    std::string baseFile=pngfilenames.substr(pattern-18,17);
    double baseImageTime=atof(baseFile.c_str());
    return baseImageTime;
}


std::vector<std::pair<std::string, std::string>> CameraCameraSync::imageTimeStampSyncFunction()
{
    std::vector<std::pair<std::string,std::string>> syncPairLists;
    double timeDifference;
    for (auto baseFileNames:oriImageLists_)
    {
        double maxSSIM=0;
        std::string anchorFileNames;
        std::cout<<"png"<<std::endl;
        double baseImageTime=getBaseTime(baseFileNames,"png");
        for (auto candidateFileNames:dstImageLists_)
        {
            double candidateImageTime=getBaseTime(candidateFileNames,"png");
            timeDifference=std::abs(baseImageTime-candidateImageTime);
            if(timeDifference<=0.1)
            {
                std::cout<<"imread"<<std::endl;
                cv::Mat orgImage=cv::imread(baseFileNames,cv::IMREAD_GRAYSCALE);
                cv::Mat dstImage=cv::imread(candidateFileNames,cv::IMREAD_GRAYSCALE);
                if(!orgImage.data&&!dstImage.data)
                {
                    std::cout<<"error read images"<<std::endl;
                    break;
                }
                double ssim=evaluateImageTimeStampSync(orgImage,dstImage);
                if (ssim>maxSSIM)
                {
                    maxSSIM=ssim;
                    anchorFileNames=candidateFileNames;
                }
            }
        }
        std::pair<std::string,std::string> syncPair(std::make_pair(baseFileNames,anchorFileNames));
        syncPairLists.push_back(syncPair);
    }
    return syncPairLists;
}

double CameraCameraSync::evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage)
{
    //这里采用SSIM结构相似性来作为图像相似性评判
    double C1 = 6.5025, C2 = 58.5225;
    int width = orgImage.cols;
    int height = orgImage.rows;

    double mean_x = 0;
    double mean_y = 0;
    double sigma_x = 0;
    double sigma_y = 0;
    double sigma_xy = 0;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            mean_x += orgImage.at<uchar>(v, u);
            mean_y += dstImage.at<uchar>(v, u);

        }
    }
    mean_x = mean_x / width / height;
    mean_y = mean_y / width / height;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            sigma_x += (orgImage.at<uchar>(v, u) - mean_x)* (orgImage.at<uchar>(v, u) - mean_x);
            sigma_y += (dstImage.at<uchar>(v, u) - mean_y)* (dstImage.at<uchar>(v, u) - mean_y);
            sigma_xy += std::abs((orgImage.at<uchar>(v, u) - mean_x)* (dstImage.at<uchar>(v, u) - mean_y));
        }
    }
    sigma_x = sigma_x / (width*height - 1);
    sigma_y = sigma_y / (width*height - 1);
    sigma_xy = sigma_xy / (width*height - 1);
    double molecule = (2 * mean_x*mean_y + C1) * (2 * sigma_xy + C2);
    double denominator = (mean_x*mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
    double ssim = molecule / denominator;
    return ssim;
}

void CameraCameraSync::spatialSynchronization(cv::Mat image1, cv::Mat image2)
{
    int minHessian=400;
    std::vector<cv::KeyPoint> keypoints_object,keypoints_scene;
    cv::Mat descriptors_object,descriptors_scene;
    cv::Ptr<cv::ORB> orb=cv::ORB::create(500,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);

    orb->detect(image1,keypoints_object);
    orb->detect(image2,keypoints_scene);

    orb->compute(image1,keypoints_object,descriptors_object);
    orb->compute(image2,keypoints_scene,descriptors_scene);

    std::vector<cv::DMatch> matchers;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_object,descriptors_scene,matchers);

    double min_dist=10000,max_dist=0;
    for (int i=0;i<descriptors_object.rows;i++)
    {
        double dist=matchers[i].distance;
        if(dist<min_dist) min_dist=dist;
        if(dist>max_dist) max_dist=dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (int i=0;i<descriptors_object.rows;i++)
    {
        if(matchers[i].distance<=std::max(2*min_dist,30.0))
        {
            good_matches.push_back(matchers[i]);
        }
    }

    cv::Mat img_match;
    cv::drawMatches(image1,keypoints_object,image2,keypoints_scene,good_matches,img_match);


    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (unsigned int i=0;i<good_matches.size();i++)
    {
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    cv::Mat H=cv::findHomography(obj,scene,cv::RANSAC);
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0]=cv::Point (0,0);
    obj_corners[1]=cv::Point (image1.cols,0);
    obj_corners[2]=cv::Point (0,image1.rows);
    obj_corners[3]=cv::Point (image1.cols,image1.rows);
    std::vector<cv::Point2f> scene_corner(4);

    cv::perspectiveTransform(obj_corners,scene_corner,H);

    time_t  timep;
    time(&timep);

    char name[1024];
    sprintf(name,"/home/melody/data/fusion/效果_%d.jpg",timep);
    cv::imwrite(name,img_match);

}