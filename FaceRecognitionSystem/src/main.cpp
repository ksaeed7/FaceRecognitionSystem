#include <iostream>
#include <string>
#include <chrono>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>
#include <opencv2/highgui.hpp>
#include "faceRecognition.h"
#include "videostreamer/videoStreamer.h"
#include "network/network.h"
#include "network/mtcnn.h"
#include "gpio/GpioControl.h"
#include <fstream>
#include "server/ServerSocket.h"
#include <condition_variable>
#include <queue>
#include "frameBuffer/SafeQueue.h"

#ifndef HIGH
#define HIGH 1
#endif

#ifndef LOW
#define LOW 0
#endif


//#include <dos.h>

using namespace std;
/*ofstream fileValue; // using GPIO pin 24
ofstream web_streamer;
ifstream motionSensorFile; // using GPIO pin pin 40
ifstream keyPressedFile; //using GPIO pin 77 which is 38
ifstream keyPinFile; //using GPIO pin 51 which is 36
ofstream keyPadOutFile; //using GPIO pin 12 which is 37 on board
ifstream proximityFile; //using GPIO pin 13 which is 22 on board
// Uncomment to print timings in milliseconds
// #define LOG_TIMES*/

using namespace nvinfer1;
using namespace nvuffparser;


//std::atomic<bool> keepRunning(true);

/****
 * 
 * Motion sensor stuff
 * 
 * 
*/
std::mutex motionMutex;
std::condition_variable motionCondition;
std::atomic<bool> motionDetected(false);
std::atomic<bool> keepRunning(true);
std::chrono::milliseconds debounceDelay(5000);

/**
 * 
 * 
 * Solenoid
 * 
*/

std::atomic<bool> solenoidActivated(false);
std::mutex solenoidMutex;
std::condition_variable solenoidCondition;
std::chrono::milliseconds solenoidDelay(2000); // Solenoid stays high for 2 seconds
/*
template<typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex m;
    std::condition_variable cv;
    bool shutdown = false;

public:
    void enqueue(T item) {
        std::lock_guard<std::mutex> lock(m);
        if(shutdown) return;
        queue.push(std::move(item));
        cv.notify_one();
    }

    T dequeue() {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return !queue.empty(); });
        //if(shutdown) return;
        T item = std::move(queue.front());
        queue.pop();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(m);
        return queue.empty();
    }
    void signalShutdown() {
        std::lock_guard<std::mutex> lock(m);
        shutdown = true;
        cv.notify_all();
    }
};*/
/*

void solenoidControlThread(GpioControl& solenoid) {
    std::unique_lock<std::mutex> lock(solenoidMutex, std::defer_lock);

    while (keepRunning) {
        lock.lock();
        solenoidCondition.wait(lock, []{ return solenoidActivated.load() || !keepRunning; });

        if (solenoidActivated) {
            solenoid.writePin(HIGH); // Activate solenoid
            std::this_thread::sleep_for(solenoidDelay); // Maintain solenoid activation for the delay duration
            solenoid.writePin(LOW); // Deactivate solenoid
            solenoidActivated = false; // Reset solenoid activation flag
        }
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to prevent tight loop when idle
        solenoid.writePin(LOW);
    }
}
*/
std::mutex faceNetMutex; // Global mutex to protect FaceRecognition instance

void keyboardThreadFunc(FaceRecognition& faceNet, mtcnn& mtCNN, SafeQueue<cv::Mat>& frameQueue) {
    char key;
    cv::Mat frame;

    while (keepRunning) {
        std::cin >> key;

        if (key == 'q' || key == 27) { // Exit program
            keepRunning = false;
            //shutdown();
            //videoStreamer.~VideoStreamer(); // Consider safely stopping the streamer instead of destructing
        }
        else if (key == 'n') {
            std::lock_guard<std::mutex> lock(faceNetMutex);
            // Ensure frame capture and face addition is thread-safe
            frame = frameQueue.dequeue();
            if (!frame.empty()) {
                auto outputBbox = mtCNN.findFace(frame);
                faceNet.addNewFace(frame, outputBbox);
                // Consider handling UI updates or logging here
            }
        }
        else if (key == 'd') {
            std::lock_guard<std::mutex> lock(faceNetMutex);
            // Safely call removeFace method
            faceNet.removeFace();
            // Consider handling UI updates or logging here
        }

        // Add more cases as needed
    }
}

void processFrames(FaceRecognition& faceNet, mtcnn& detector,/*,GpioControl& solenoid*/ SafeQueue<cv::Mat>& frameQueue) {
    int facesDetected = 0; // Track the number of faces detected
    int recognizeFace = 0; // This will be set if a face is recognized
    std::cout<<"Processessing faces thread starting"<<std::endl;
    std::cout << "Processing faces thread starting" << std::endl;
    bool processFaces = false;
    auto processUntil = std::chrono::steady_clock::now();

    while (keepRunning) {
        //cv::Mat frame;

        // Check if we are in the face processing window
        if (std::chrono::steady_clock::now() < processUntil) {
            processFaces = true;
        } else {
            std::unique_lock<std::mutex> lock(motionMutex);
            motionCondition.wait_for(lock, std::chrono::milliseconds(10), []{
                return motionDetected.load() || !keepRunning;
            });
            if (motionDetected.load()) {
                // Set the time window for processing faces after motion detection
                processUntil = std::chrono::steady_clock::now() + debounceDelay;
                processFaces = true;
                motionDetected = false; // Reset motion detected to await next detection
            } else {
                processFaces = false;
            }
        }

        //auto processUntil = std::chrono::steady_clock::now() + debounceDelay;

        
            if (!frameQueue.empty()) 
            {
                cv::Mat frame = frameQueue.dequeue();

                if (!frame.empty() && processFaces) 
                {
                    //std::vector<struct Bbox> detectedFaces = detector.findFace(frame);
                    //std::cout<<"Prelock Queue"<<std::endl;

                    std::lock_guard<std::mutex> guard(faceNetMutex); // Lock the mutex to safely use faceNet
                    // Reset recognition state variables for each frame
                    faceNet.resetVariables();

                    //auto startMTCNN = std::chrono::steady_clock::now();
                    std::vector<struct Bbox> outputBbox = detector.findFace(frame);
                    //auto endMTCNN = std::chrono::steady_clock::now();

                    //int facesDetected;
                    // Forward faces to FaceNet for recognition
                    //auto startForward = std::chrono::steady_clock::now();
                    faceNet.forward(frame, outputBbox, facesDetected); // Assume it modifies 'frame' to mark recognized faces
                   // auto endForward = std::chrono::steady_clock::now();

                    #ifdef LOG
                        std::cout << "MTCNN Detection Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endMTCNN - startMTCNN).count() << "ms\n";
                        std::cout << "FaceNet Inference Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endForward - startForward).count() << "ms\n";
                    #endif

                    // Check if any faces were recognized, and get recognition results
                    if (facesDetected > 0)
                    {
                        // Process recognized faces
                        faceNet.featureMatching(frame, recognizeFace);

                        //Update solenoid sensor interrupts:
                        //std::lock_guard<std::mutex> guard(solenoidMutex);
                        //solenoidActivated = true;
                        //solenoidCondition.notify_one(); // Signal the solenoid thread

                        // Optionally: Display the frame with annotations
                        
                    }
                    
                }
                cv::imshow("Recognized Faces", frame);
                cv::waitKey(1); // Use waitKey(1) to display the frame; adjust as needed.
            }
        
        motionDetected = false;

    }
}


void captureThreadFunc(VideoStreamer& videoStreamer, SafeQueue<cv::Mat>& frameQueue) {
    cv::Mat frame;
    while (keepRunning) {
        videoStreamer.getFrame(frame);
        if (!frame.empty()) {
            frameQueue.enqueue(frame.clone());
        }
    }
}

/*
std::mutex motionMutex;
std::atomic<bool> motionDetected(false);
std::condition_variable motionCondition;
std::chrono::milliseconds debounceDelay(1500); // Example debounce delay of 500ms*/

void motionSensorThreadFunc() {
    GpioControl motionSensor(38); // Example GPIO pin
    motionSensor.setDirection("in");
    bool lastState = false;
    auto lastDebounceTime = std::chrono::steady_clock::now();

    while (keepRunning) {
        bool currentState = motionSensor.readPin() == HIGH;

        // Check for a state change
        if (currentState != lastState) {
            lastDebounceTime = std::chrono::steady_clock::now();
            //std::cout<<"Motion detected"<<std::endl;
            motionDetected.store(true);
        }

        // If the current state has been stable for longer than the debounce delay, take action
        if ((std::chrono::steady_clock::now() - lastDebounceTime) > debounceDelay) {
            // If motion is detected and has been stable, set the motionDetected flag
            if (currentState && !motionDetected) {
                
                //motionDetected = true;
                motionDetected.store(true);
                // Optional: Log or take other actions upon motion detection
                std::cout << "Motion detected." << std::endl;
            }
        }

        lastState = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Poll every 50ms
    }
}
/*
void motionSensorThread() {
    GpioControl motionSensor(78); // Adjust for your specific GPIO pin
    while (keepRunning) {
        if (motionSensor.readPin() == HIGH) {
            {
                std::lock_guard<std::mutex> lock(motionMutex);
                motionDetected = true;
            }
            motionCondition.notify_one();
            std::this_thread::sleep_for(debounceDelay); // Wait for the debounce period before checking for motion again
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Check every 50ms
    }
}*/


void initializeFaces(int maxFacesPerScene, int videoFrameWidth,int videoFrameHeight, FaceRecognition &faceNet, mtcnn &mtCNN)
{
    std::vector<struct Bbox> outputBbox;
    outputBbox.reserve(maxFacesPerScene);
    std::vector<struct Paths> paths;
    cv::Mat image;
    getFilePaths("../imgs", paths);
    for(int i=0; i < paths.size(); i++) {
        loadInputImage(paths[i].absPath, image, videoFrameWidth, videoFrameHeight);
        outputBbox = mtCNN.findFace(image);
        std::size_t index = paths[i].fileName.find_last_of(".");
        std::string rawName = paths[i].fileName.substr(0,index);
        faceNet.forwardAddFace(image, outputBbox, rawName);
        faceNet.resetVariables();
    }
    outputBbox.clear();

}
void shutdown(SafeQueue<cv::Mat>& frameQueue) {
    keepRunning = false;

    // Signal all condition variables
    motionCondition.notify_all();
    solenoidCondition.notify_all();

    // Signal the SafeQueue to release any waiting threads
    frameQueue.signalShutdown();
}


int main()
{
    Logger gLogger;
    if (!initLibNvInferPlugins(&gLogger, "")) {
        std::cerr << "Failed to initialize TensorRT plugins." << std::endl;
        return 1;
    }
    DataType dtype = DataType::kHALF;
    const std::string uffFile = "../google_facenet_models/facenet.uff";
    const std::string engineFile = "../google_facenet_models/facenet.engine";
    bool serializeEngine = true;
    int batchSize = 1;
    int videoFrameWidth = 640;
    int videoFrameHeight = 480;
    int maxFacesPerScene = 2;
    float knownPersonThreshold = 1.0;

    // Initialize FaceNet and MTCNN
    FaceRecognition faceNet(gLogger, dtype, uffFile, engineFile, batchSize, serializeEngine, knownPersonThreshold, maxFacesPerScene, videoFrameWidth, videoFrameHeight);
    mtcnn mtCNN(videoFrameHeight, videoFrameWidth);

    // Initialize Video Streamer
    //VideoStreamer videoStreamer("0", videoFrameWidth, videoFrameHeight); // Assuming "0" is a valid identifier for your camera


    initializeFaces(maxFacesPerScene, videoFrameWidth, videoFrameHeight, faceNet, mtCNN);
    // Initialize GPIO Control for Solenoid
    GpioControl solenoid(19); // Adjust GPIO pin number as necessary

    // Initialize SafeQueue for frames
    SafeQueue<cv::Mat> frameQueue;

    ServerSocket server(55047);

    server.startServer(frameQueue);
    std::unique_lock<std::mutex> lk(server.mtx_);
    server.cv_.wait(lk, [&]{ return true;}); // isClientConnected() should check for a valid clientSocket

// Now it's safe to handle the client since one has been connected


    cout << "FOUND CLIENT" <<endl;
    //std::thread handleClientThread(handleClientConnection, server.clientSocket, std::ref(frameQueue));
    // Start threads
    //std::thread solenoidThread(solenoidControlThread, std::ref(solenoid));
    std::thread motionSensorThread(motionSensorThreadFunc);
    //std::thread captureThread(captureThreadFunc, std::ref(videoStreamer), std::ref(frameQueue));
    std::thread frameProcessingThread(processFrames, std::ref(faceNet), std::ref(mtCNN), std::ref(frameQueue));
    std::thread keyboardThread(keyboardThreadFunc, std::ref(faceNet), std::ref(mtCNN), std::ref(frameQueue));


    //handleClientThread.join();

    // Wait for threads to finish
    //solenoidThread.join();
    motionSensorThread.join();
    //handleClientThread.join();
    //captureThread.join();
    frameProcessingThread.join();
    keyboardThread.join();
    

    return 0;
}
