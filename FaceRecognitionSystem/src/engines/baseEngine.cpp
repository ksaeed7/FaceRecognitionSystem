// Include guard for the header file to prevent multiple inclusions
#include "baseEngine.h"

// Static member initialization for det1_relu_counter
int baseEngine::det1_relu_counter = 1;

// Constructor: Initializes the baseEngine object with model paths and input/output blob names
baseEngine::baseEngine(const char * prototxt, const char* model, const char* input_name, const char* location_name,
                       const char* prob_name, const char *point_name) :
                             prototxt(prototxt),
                             model(model),
                             INPUT_BLOB_NAME(input_name),
                             OUTPUT_LOCATION_NAME(location_name),
                             OUTPUT_PROB_NAME(prob_name),
                             OUTPUT_POINT_NAME(point_name)
{
};

// Destructor: Handles any required cleanup, specifically for the Protobuf library used by TensorRT
baseEngine::~baseEngine() {
    shutdownProtobufLibrary();
}

// Placeholder for initialization method, potentially for initializing resources or pre-processing steps
void baseEngine::init(int row, int col) {
}

// Converts a Caffe model to a TensorRT engine, allowing for optimization and faster inference
void baseEngine::caffeToGIEModel(const std::string &deployFile,                // Caffe prototxt file path
                                 const std::string &modelFile,                // Caffe model file path
                                 const std::vector<std::string> &outputs,   // Names of output tensors
                                 unsigned int maxBatchSize,                  // Maximum batch size for inference
                                 IHostMemory *&gieModelStream)              // Output pointer to the serialized TensorRT engine
{
    // Logic to generate a unique name for the TensorRT engine file, based on the input model name
    size_t lastIdx = model.find_last_of(".");
    string enginePath = model.substr(0, lastIdx);
    if(enginePath.find("det1_relu") != std::string::npos) {
        enginePath.append(std::to_string(det1_relu_counter));
        enginePath.append(".engine");
        det1_relu_counter++;
    }
    else {
        enginePath.append(".engine");
    }
    std::cout << "rawName = " << enginePath << std::endl;

    // If the engine file already exists, deserialize it and set up the execution context
    if(fileExists(enginePath)) {
        std::vector<char> trtModelStream_;
        size_t size{ 0 };

        std::ifstream file(enginePath, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream_.resize(size);
            file.read(trtModelStream_.data(), size);
            file.close();
        }
        std::cout << "size" << size;
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream_.data(), size, nullptr);
        assert(engine);
        context = engine->createExecutionContext();
    }
    else {
        // If the engine file does not exist, build it from the Caffe model
        IBuilder *builder = createInferBuilder(gLogger);
        INetworkDefinition *network = builder->createNetwork();
        ICaffeParser *parser = createCaffeParser();

        // Parsing the Caffe model and marking outputs
        const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
                                                                  modelFile.c_str(),
                                                                  *network,
                                                                  nvinfer1::DataType::kHALF);
        for (auto &s : outputs)
            network->markOutput(*blobNameToTensor->find(s.c_str()));

        // Configuring and building the TensorRT engine
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(1 << 25);
        ICudaEngine *engine = builder->buildCudaEngine(*network);
        assert(engine);

        context = engine->createExecutionContext();

        // Serializing the engine for storage
        ofstream planFile;
        planFile.open(enginePath);
        IHostMemory *serializedEngine = engine->serialize();
        planFile.write((char *) serializedEngine->data(), serializedEngine->size());
        planFile.close();

        // Cleaning up
        network->destroy();
        parser->destroy();
        builder->destroy();
    }
}
