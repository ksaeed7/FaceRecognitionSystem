#ifndef MTCNN_H
#define MTCNN_H
#include "../network/network.h"
#include "../engines/pnet_rt.h"
#include "../engines/rnet_rt.h"
#include "../engines/onet_rt.h"
class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    vector<struct Bbox> findFace(cv::Mat &image);
private:
    cv::Mat reImage;
    float nms_threshold[3];
    vector<float> scales_;
    Pnet_engine *pnet_engine;
    Pnet **simpleFace_;
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;
    Rnet *refineNet;
    Rnet_engine *rnet_engine;
    vector<struct Bbox> secondBbox_;
    vector<struct orderScore> secondBboxScore_;
    Onet *outNet;
    Onet_engine *onet_engine;
    vector<struct Bbox> thirdBbox_;
    vector<struct orderScore> thirdBboxScore_;
};

#endif