# Face Recognition Security System

A comprehensive facial recognition system designed to integrate seamlessly with IoT devices, leveraging deep learning for accurate identification and hardware control for real-time interaction. This system uses MTCNN for face detection and a google FaceNet implementation for face recognition, interfaced with hardware controls via GPIO for various applications, including access control and monitoring.

## Prerequisites

Before you begin, ensure you meet the following requirements:
- **Operating System**: Linux-based OS (Ubuntu recommended) with GPIO support if utilizing hardware features.
- **Hardware**: NVIDIA GPU with CUDA® Compute Capability 5.2 or later for TensorRT optimizations.
- **Software**:
  - CUDA 10.2 or later
  - cuDNN 7.6.5 or later
  - TensorRT 7.0 or later
  - OpenCV 4.1 or later
  - CMake 3.8 or later
  - A C++ compiler compatible with C++14

## Installation

To set up the project environment, follow these steps:

- JetsonNano should already have mostly everything installed. You would need cmake, openblas, and tensorflow.

```bash
sudo apt install cmake libopenblas-dev
```

- Installing tensorflow Nano:
```bash
# Install system packages required by TensorFlow:
sudo apt update
sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# Install and upgrade pip3
sudo apt install python3-pip
sudo pip3 install -U pip testresources setuptools

# Install the Python package dependencies
sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

# Install TensorFlow using the pip3 command. This command will install the latest version of TensorFlow compatible with JetPack 4.4.
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'
```

### Model can be pruned and frozen:
 - Use this: https://github.com/apollo-time/facenet/raw/master/model/resnet/facenet.pb
 - run convert_pb_to_uff.py

### Refer to PKUZHOU for mtCNN
 - I uploaded the mtCNN models for easy use. 
 - But you can always do it like so:

```bash
# go to one above project,
cd path/to/project/..
# clone PKUZHOUs repo,
git clone https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT
# and move models into mtCNNModels folder
mv MTCNN_FaceDetection_TensorRT/det* path/to/project/mtCNNModels
```

### If you want to simple download and build repo:

1. **CUDA & cuDNN**: Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) according to your system setup.
2. **TensorRT**: Follow the [official guide](https://developer.nvidia.com/tensorrt) to install TensorRT.
3. **OpenCV**: Install OpenCV using your system's package manager or build from source following [this tutorial](https://opencv.org/releases/).
4. **Clone the repository**:
   ```bash
   git clone https://github.com/ksaeed7/FaceRecognitionSystem
   cd FaceRecognitionSystem
    ```
5. **Build the project**
    ```bash
   mkdir build && cd build
    cmake ..
    make 
    ```

## Usage 

 - To run the facial recognition system, follow these steps:

1. **Start the Server:**
    ```bash
    ./FaceRecognitionSystem
    ```

2. **Connect Client**
    
 - You can use any client, just ensure you link ip_address and same port.
 - I used an ESP32-CAM to stream the image data via WiFi, you can use whatever you like.


## More details


## System Architecture and Performance

### Multithreading and Concurrency
The system is designed to leverage multi-threading to enhance performance and responsiveness. Using C++ standard threading, the application efficiently manages multiple tasks in parallel, including video frame capture, face detection and recognition processing, and server-client communication. Key threading components include:


- **Face Processing Thread**: Dedicated to processing video frames for face detection and recognition. This thread works independently of the video capture, allowing for uninterrupted frame acquisition and using GPU for processing.
- **Motion Detection Thread**: Monitors GPIO inputs for motion detection signals. Upon detecting motion, it triggers the face recognition process, providing a smart start mechanism.
- **Server Communication Thread**: Manages client connections and data transmission, ensuring smooth video streaming from clients for processing.
- **User Interaction Thread**: Listens for keyboard inputs, allowing users to dynamically add or remove faces from the database or to terminate the application.
- **FrameBuffer Thread**: Continuously captures video frames from the server camera, ensuring real-time processing without blocking the main application flow.

This multi-threaded approach ensures that the system can handle intensive tasks without performance degradation, making efficient use of the CPU and GPU resources.

### Utilizing Jetson Nano GPU Cores
The system is optimized for the NVIDIA Jetson Nano, harnessing its GPU cores to accelerate the face detection and recognition process. By leveraging the CUDA-enabled GPU, the system can perform complex computations required for real-time face recognition more efficiently than CPU-based processing. Key aspects include:

- **TensorRT Optimization**: The FaceNet and MTCNN models are optimized with TensorRT, significantly improving inference times by taking full advantage of the Jetson Nano's GPU architecture.
- **Parallel Processing**: The CUDA cores enable parallel processing of video frames and neural network operations, allowing for rapid face detection and recognition even in high-throughput scenarios.
- **Efficient Resource Management**: The application is designed to maximize GPU utilization while minimizing memory overhead, ensuring smooth operation and real-time responsiveness.

### Integration with C++ Threading
The application harnesses C++11 threading capabilities to manage its multi-component architecture. Threads are used not just for processing tasks in parallel but also for synchronizing operations such as frame capture and processing, user interactions, and hardware control signals. Proper synchronization mechanisms, including mutexes and condition variables, are employed to ensure data consistency and thread safety across the system.

### Server-Client Communication
- The system starts by initializing a server listening for incoming connections.
- Clients can connect to this server to stream video frames to be processed for facial recognition.
- To save resources, facial recognition is triggered upon motion detection.

### Keyboard Interactions
- **Quit Program**: Press `'q'` or the **ESC** key to gracefully shutdown the server and all processing threads, terminating the program.
- **Add New Face**: Press `'n'` to capture the current frame from the video stream and add a new face to the recognition database.
- **Delete Face**: Press `'d'` to remove a face from the recognition database. There must be a face present in the frame. The specific method of selecting which face to remove is assumed to be implemented.

### Hardware-Specific Controls
- **Motion Sensors**: Utilize motion sensors to detect the presence of individuals. The system can be configured to activate face recognition upon motion detection.
- **Solenoid Locks**: Based on the recognition result, solenoid locks can be triggered to unlock, allowing access if a recognized face is authorized.
- **Keypad**: Pressing the keypad, locks the mutex for face recognition and frame capture until time delay or until pin is entered and door is unlocked.

## Software Functions

The main function orchestrates various components of the system, establishing the groundwork for server-client communication, video streaming, and interaction through hardware controls and keyboard inputs.

### Initialization
- Sets up necessary components such as GPIO pins for motion detection and solenoid control, loading the facial recognition model, and running the engine.

### Frame Capture
- There are two options, you can simply plug a camera and update code to add frames from openCV video streamer to the queue keeping the components local.
- Or run the client on another hardware that sends the image data to devices specific address and port. Ensure you are updating the ip address and port accordingly.

### Face Detection and Recognition
- Integrates MTCNN for face detection within video frames and utilizes a customized FaceNet model for recognizing faces. The system can add or identify known individuals in real-time. 

### Motion Detection
- Uses GPIO-based motion sensors to activate the face recognition process efficiently, conserving computational resources when no presence is detected.

### Solenoid Control
- The system can trigger solenoids to unlock doors or gates based on face recognition results, providing a mechanism for physical access control.

### User Input Handling
- The system allows users to add or remove faces from the recognition database or terminate the program through simple keyboard commands.

This integration of video processing, facial recognition, and hardware control offers a comprehensive solution for security and monitoring applications, allowing users to interact with the system in real-time through server-client communication, direct keyboard commands, and responsive hardware interactions.

### Key Functions

- **initializeFaces():** Preloads known faces to speed up the recognition process.
- **processFrames():** The core loop for processing video frames, detecting and recognizing faces, and controlling hardware based on recognition results.
- **captureThreadFunc():** A dedicated thread for capturing video frames and enqueuing them for processing.
- **motionSensorThreadFunc():** Monitors GPIO for motion detection, triggering the facial recognition process.


