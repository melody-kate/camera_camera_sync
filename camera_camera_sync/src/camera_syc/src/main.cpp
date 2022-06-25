//
// Created by melody on 2022/6/16.
//
#include "camera_camera_sync.h"

int main()
{
    CameraCameraSync cameraCameraSync;
    std::string oriDirs="/home/melody/data/practice_1_1_multi_camera_sync/camera_front_left_60";
    std::string dstDirs="/home/melody/data/practice_1_1_multi_camera_sync/camera_front_right_60";
    cameraCameraSync.getImageTimeStamp(oriDirs,dstDirs);

    std::vector<std::pair<std::string,std::string>> syncImageLists;
    int number=cameraCameraSync.getImageNumber();
    if (number>0)
    {
        syncImageLists=cameraCameraSync.imageTimeStampSyncFunction();
    }
    for (auto syncPair:syncImageLists)
    {
        cv::Mat image1=cv::imread(syncPair.first,cv::IMREAD_GRAYSCALE);
        cv::Mat image2=cv::imread(syncPair.second,cv::IMREAD_GRAYSCALE);
        if( !image1.data || !image2.data )
        {
            std::cout<< " --(!) Error reading images " << std::endl;
            return -1;
        }

        cameraCameraSync.spatialSynchronization(image1, image2);
    }



    return 0;
}
