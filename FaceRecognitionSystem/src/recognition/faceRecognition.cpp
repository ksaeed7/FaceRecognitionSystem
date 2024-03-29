#include "faceRecognition.h"
int countFilter = 0;

int FaceRecognition::m_classCount = 0;

FaceRecognition::FaceRecognition
(Logger gLogger, DataType dtype, const string uffFile, const string engineFile, int batchSize, bool serializeEngine,
        float knownPersonThreshold, int maxFacesPerScene, int frameWidth, int frameHeight) {

    m_INPUT_C = static_cast<const int>(3);
    m_INPUT_H = static_cast<const int>(160);
    m_INPUT_W = static_cast<const int>(160);
    m_frameWidth = static_cast<const int>(frameWidth);
    m_frameHeight = static_cast<const int>(frameHeight);
    m_gLogger = gLogger;
    m_dtype = dtype;
    m_uffFile = static_cast<const string>(uffFile);
    m_engineFile = static_cast<const string>(engineFile);
    m_batchSize = batchSize;
    m_serializeEngine = serializeEngine;
    m_maxFacesPerScene = maxFacesPerScene;
    m_croppedFaces.reserve(maxFacesPerScene);
    m_embeddings.reserve(128);
    m_knownPersonThresh = knownPersonThreshold;
    cout<<"\Known person threshold: "<< knownPersonThreshold;
    // load engine from .engine file or create new engine
    this->createOrLoadEngine();
}


void FaceRecognition::createOrLoadEngine() {
    if(fileExists(m_engineFile)) {
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(m_engineFile, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            std::cout << "size" << trtModelStream_.size() << std::endl;
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        std::cout << "size" << size;
        IRuntime* runtime = createInferRuntime(m_gLogger);
        assert(runtime != nullptr);
        m_engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        std::cout << std::endl;
    }
    else {
        IBuilder *builder = createInferBuilder(m_gLogger);
        INetworkDefinition *network = builder->createNetwork();
        IUffParser *parser = createUffParser();
        parser->registerInput("input", DimsCHW(160, 160, 3), UffInputOrder::kNHWC);
        parser->registerOutput("embeddings");

        if (!parser->parse(m_uffFile.c_str(), *network, m_dtype))
        {
            cout << "Failed to parse UFF\n";
            builder->destroy();
            parser->destroy();
            network->destroy();
            throw std::exception();
        }

        /* build engine */
        if (m_dtype == DataType::kHALF)
        {
            builder->setFp16Mode(true);
        }
        else if (m_dtype == DataType::kINT8) {
            builder->setInt8Mode(true);
            // ToDo
            //builder->setInt8Calibrator()
        }
        builder->setMaxBatchSize(m_batchSize);
        builder->setMaxWorkspaceSize(1<<30);
        // strict will force selected datatype, even when another was faster
        //builder->setStrictTypeConstraints(true);
        // Disable DLA, because many layers are still not supported
        // and this causes additional latency.
        //builder->allowGPUFallback(true);
        //builder->setDefaultDeviceType(DeviceType::kDLA);
        //builder->setDLACore(1);
        m_engine = builder->buildCudaEngine(*network);

        /* serialize engine and write to file */
        if(m_serializeEngine) {
            ofstream planFile;
            planFile.open(m_engineFile);
            IHostMemory *serializedEngine = m_engine->serialize();
            planFile.write((char *) serializedEngine->data(), serializedEngine->size());
            planFile.close();
        }

        /* break down */
        builder->destroy();
        parser->destroy();
        network->destroy();
    }
    m_context = m_engine->createExecutionContext();
}


void FaceRecognition::getCroppedFacesAndAlign(cv::Mat frame, std::vector<struct Bbox> outputBbox) {
    for(vector<struct Bbox>::iterator it=outputBbox.begin(); it!=outputBbox.end();it++){
        if((*it).exist){
            cv::Rect facePos(cv::Point((*it).y1, (*it).x1), cv::Point((*it).y2, (*it).x2));
            cv::Mat tempCrop = frame(facePos);
            struct CroppedFace currFace;
            cv::resize(tempCrop, currFace.faceMat, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
            currFace.x1 = it->x1;
            currFace.y1 = it->y1;
            currFace.x2 = it->x2;
            currFace.y2 = it->y2;            
            m_croppedFaces.push_back(currFace);
        }
    }
    //ToDo align
}

void FaceRecognition::preprocessFaces() {
    // preprocess according to facenet training and flatten for input to runtime engine
    for (int i = 0; i < m_croppedFaces.size(); i++) {
        //mean and std
        cv::cvtColor(m_croppedFaces[i].faceMat, m_croppedFaces[i].faceMat, cv::COLOR_RGB2BGR);
        cv::Mat temp = m_croppedFaces[i].faceMat.reshape(1, m_croppedFaces[i].faceMat.rows * 3);
        cv::Mat mean3;
        cv::Mat stddev3;
        cv::meanStdDev(temp, mean3, stddev3);

        double mean_pxl = mean3.at<double>(0);
        double stddev_pxl = stddev3.at<double>(0);
        cv::Mat image2;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_64FC1);
        m_croppedFaces[i].faceMat = image2;
        // fix by peererror
        cv::Mat mat(4, 1, CV_64FC1);
		mat.at <double>(0, 0) = mean_pxl;
		mat.at <double>(1, 0) = mean_pxl;
		mat.at <double>(2, 0) = mean_pxl;
		mat.at <double>(3, 0) = 0;
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat - mat;
        // end fix
        m_croppedFaces[i].faceMat = m_croppedFaces[i].faceMat / stddev_pxl;
        m_croppedFaces[i].faceMat.convertTo(image2, CV_32FC3);
        m_croppedFaces[i].faceMat = image2;
    }
}


void FaceRecognition::doInference(float* inputData, float* output) {
    int size_of_single_input = 3 * 160 * 160 * sizeof(float);
    int size_of_single_output = 128 * sizeof(float);
    int inputIndex = m_engine->getBindingIndex("input");
    int outputIndex = m_engine->getBindingIndex("embeddings");

    void* buffers[2];

    cudaMalloc(&buffers[inputIndex], m_batchSize * size_of_single_input);
    cudaMalloc(&buffers[outputIndex], m_batchSize * size_of_single_output);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // copy data to GPU and execute
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, m_batchSize * size_of_single_input, cudaMemcpyHostToDevice, stream));
    m_context->enqueue(m_batchSize, &buffers[0], stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], m_batchSize * size_of_single_output, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


void FaceRecognition::forwardAddFace(cv::Mat image, std::vector<struct Bbox> outputBbox,
        const string className) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    getCroppedFacesAndAlign(image, outputBbox);
    if(!m_croppedFaces.empty()) {
        preprocessFaces();
        doInference((float*)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();
}

void FaceRecognition::forward(cv::Mat frame, std::vector<struct Bbox> outputBbox, int &facesDetected) {
    getCroppedFacesAndAlign(frame, outputBbox); // ToDo align faces according to points
    preprocessFaces();
    for(int i = 0; i < m_croppedFaces.size(); i++) {
        doInference((float*)m_croppedFaces[i].faceMat.ptr<float>(0), m_output);
        m_embeddings.insert(m_embeddings.end(), m_output, m_output+128);
    }
    //new code here for "checking if face is detected"
    if(m_croppedFaces.empty())
    {
        alreadyRecognized = false;
        facesDetected=0;
    }
    else
        facesDetected=1;
}

void FaceRecognition::featureMatching(cv::Mat &image, int &recognizeFace) {
	//new code
	
			
    for(int i = 0; i < (m_embeddings.size()/128); i++) {
        double minDistance = 10.* m_knownPersonThresh;
        float currDistance = 0.;
        int winner = -1;
        for (int j = 0; j < m_knownFaces.size(); j++) {
            std:vector<float> currEmbedding(128);
            std::copy_n(m_embeddings.begin()+(i*128), 128, currEmbedding.begin());
            currDistance = vectors_distance(currEmbedding, m_knownFaces[j].embeddedFace);
            // printf("The distance to %s is %.10f \n", m_knownFaces[j].className.c_str(), currDistance);
            // if ((currDistance < m_knownPersonThresh) && (currDistance < minDistance)) {
            if (currDistance < minDistance) {
                    minDistance = currDistance;
                    winner = j;
            }
            currEmbedding.clear();
        }
        float fontScaler = static_cast<float>(m_croppedFaces[i].x2 - m_croppedFaces[i].x1)/static_cast<float>(m_frameWidth);
      //Temp // cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), 
                        //cv::Scalar(0,0,255), 2,8,0);
        if (minDistance <= m_knownPersonThresh) {
	//Send Face signal detected
        m_knownFaces[i].countFilter++;
        if(m_knownFaces[i].countFilter % 4==0){ 
        cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), 
                        cv::Scalar(0,255,0), 2,8,0);
        
		recognizeFace = 1;
        count_UnknownFaces--;
            alreadyRecognized = true;
            cv::putText(image, m_knownFaces[winner].className, cv::Point(m_croppedFaces[i].y1+2, m_croppedFaces[i].x2-3),
                    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler,  cv::Scalar(0,255,0,255), 1);
            std::cout<<"Face Recognized: "<<m_knownFaces[winner].className<<std::endl;
        }
            cv::putText(image, to_string(minDistance), cv::Point(m_croppedFaces[i].y2+2, m_croppedFaces[i].x2-3),
                    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler,  cv::Scalar(0,255,0,255), 1);
        }
        else if (minDistance > m_knownPersonThresh || winner == -1){

            cv::rectangle(image, cv::Point(m_croppedFaces[i].y1, m_croppedFaces[i].x1), cv::Point(m_croppedFaces[i].y2, m_croppedFaces[i].x2), 
                        cv::Scalar(0,0,255), 2,8,0);
            m_knownFaces[i].countFilter = 0;
            if(alreadyRecognized==false)
		    recognizeFace = 0;
            else if(count_UnknownFaces % m_croppedFaces.size()==0 )
            {
            alreadyRecognized=false;
            recognizeFace = 0;
            }
            else if(alreadyRecognized==true)
            recognizeFace=1;
            
            count_UnknownFaces++;
            cv::putText(image, "New Person", cv::Point(m_croppedFaces[i].y1+2, m_croppedFaces[i].x2-3),
                    cv::FONT_HERSHEY_DUPLEX, 0.1 + 2*fontScaler ,  cv::Scalar(0,0,255,255), 1);
        }
    }
	
}

void FaceRecognition::addNewFace(cv::Mat &image, std::vector<struct Bbox> outputBbox) {
    std::cout << "Adding new person...\nPlease make sure there is only one face in the current frame.\n"
              << "What's your name? ";
    string newName;
    std::cin >> newName;
    std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);
    string filePath = "../imgs/";
    filePath.append(newName);
    filePath.append(".jpg");
    
    cv::imwrite(filePath, image);
}
void FaceRecognition::forwardRemoveFace(
        const string className) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    //getCroppedFacesAndAlign(image, outputBbox);
    if(!m_knownFaces.empty()) {
        /*preprocessFaces();
        doInference((float*)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
        struct KnownID person;
        person.className = className;
        person.classNumber = m_classCount;
        person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);
        m_knownFaces.push_back(person);
        m_classCount++;
    }
    m_croppedFaces.clear();*/
    /*vector <int>::iterator position = find(m_knownFaces.begin(), m_knownFaces.end(),className);
    if(position != m_knownFaces.end())
    {
        cout<<className<<" was successfully removed from the system "<<endl;
        m_knownFaces.erase(position);
    }*/
    int position = -1;
    for(int i = 0; i<m_knownFaces.size();i++)
    {
        if(m_knownFaces[i].className == className)
            {
                cout<<className<<" was successfully removed from the system "<<endl;
                m_knownFaces.erase(m_knownFaces.begin()+i);
                position = i;
            }

    }
    if(position == -1)
        cout<<className<< " was not found in the system."<<endl;
    }
}

void FaceRecognition::removeFace() {
    std::cout 
              << "Who would you like to remove? ";
    string newName;
    std::cin >> newName;
   /* std::cout << "Hi " << newName << ", you will be added to the database.\n";
    forwardAddFace(image, outputBbox, newName);*/
    string filePath = "../imgs/";
    filePath.append(newName);
    filePath.append(".jpg");
    int n =filePath.length();
    char char_array[n+1];
    strcpy(char_array,filePath.c_str());
    //cv::imwrite(filePath, image);
    remove(char_array);
    forwardRemoveFace(newName);
}
void FaceRecognition::resetVariables() {
    m_embeddings.clear();
    m_croppedFaces.clear();
}

FaceRecognition::~FaceRecognition() {
    // this leads to segfault if engine or context could not be created during class instantiation
    this->m_engine->destroy();
    this->m_context->destroy();
    std::cout << "FaceNet was destructed" << std::endl;
}


// HELPER FUNCTIONS
// Computes the distance between two std::vectors
float vectors_distance(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<double>	auxiliary;
    std::transform (a.begin(), a.end(), b.begin(), std::back_inserter(auxiliary),//
                    [](float element1, float element2) {return pow((element1-element2),2);});
    auxiliary.shrink_to_fit();
    float loopSum = 0.;
    for(auto it=auxiliary.begin(); it!=auxiliary.end(); ++it) loopSum += *it;

    return  std::sqrt(loopSum);
} 



inline unsigned int elementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32:
            // Fallthrough, same as kFLOAT
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}
