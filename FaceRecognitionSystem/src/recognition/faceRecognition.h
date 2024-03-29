#ifndef FACE_RECOGNITION_FACENET_H
#define FACE_RECOGNITION_FACENET_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <numeric>
#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>
#include "../common/common.h"
#include "../engines/pBox.h"

using namespace nvinfer1;
using namespace nvuffparser;

struct CroppedFace {
    cv::Mat faceMat;
    int x1, y1, x2, y2;
};


class FaceRecognition 
{
    public:
        FaceRecognition(Logger gLogger, DataType dtype, const string uffFile, const string engineFile, int batchSize,
                bool serializeEngine, float knownPersonThreshold, int maxFacesPerScene, int frameWidth, int frameHeight);
        ~FaceRecognition();

        void createOrLoadEngine();
        void getCroppedFacesAndAlign(cv::Mat frame, std::vector<struct Bbox> outputBbox);
        void preprocessFaces();
        void doInference(float* inputData, float* output);
        void forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox, const string className);
        void forward(cv::Mat image, std::vector<struct Bbox> outputBbox, int &facesDetected);
        void featureMatching(cv::Mat &image, int &recognizeFace);
        void addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox);
        void resetVariables();
        void removeFace();
        void forwardRemoveFace( const string className);
    private:
        static int m_classCount;
        int m_INPUT_C;
        int m_INPUT_H;
        int m_INPUT_W;
        int m_frameWidth, m_frameHeight;
        Logger m_gLogger;
        DataType m_dtype;
        string m_uffFile;
        string m_engineFile;
        int m_batchSize;
        bool m_serializeEngine;
        int m_maxFacesPerScene;
        ICudaEngine *m_engine;
        IExecutionContext *m_context;
        float m_output[128];
        std::vector<float> m_embeddings;
        std::vector<struct KnownID> m_knownFaces;
        // std::vector<cv::Mat> m_croppedFaces;
        std::vector<struct CroppedFace> m_croppedFaces;
        bool alreadyRecognized = false;
        int count_UnknownFaces=0;
        float m_knownPersonThresh;
};

float vectors_distance(const std::vector<float>& a, const std::vector<float>& b);
inline unsigned int elementSize(nvinfer1::DataType t);

#endif //FACE_RECOGNITION_FACENET_H
