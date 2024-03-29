#ifndef SERVER_SOCKET_H
#define SERVER_SOCKET_H

#include <iostream>
#include <atomic>
#include <thread>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <condition_variable>
#include "../framebuffer/SafeQueue.h"
#include <queue>
#include <opencv2/highgui.hpp>



// Include your SafeQueue header here if it's used within the ServerSocket interface or public methods
// #include "SafeQueue.h"

class ServerSocket {
public:
    ServerSocket(int port);
    ~ServerSocket();

    bool startServer(SafeQueue<cv::Mat>& frameQueue);
    void stopServer();
    int clientSocket = -1;
     std::thread acceptThread_;
    std::mutex mtx_; // Mutex for synchronization
    std::condition_variable cv_; // Condition variable for synchronization
    bool isClientConnected();
    void handleClientConnection(int clientSocket, SafeQueue<cv::Mat>& frameQueue);
    // Make sure clientSocket is properly declared here
void acceptConnections(SafeQueue<cv::Mat>& frameQueue);
private:
    int port_;
    int serverFd_;
    //std::thread acceptThread_;
    std::atomic<bool> isRunning_;

    
    // Declare other private methods here if needed, such as handleClientConnection
    // void handleClientConnection(int clientSocket, SafeQueue<YourDataType>& frameQueue);

    static void log(const std::string& message);
};

#endif // SERVER_SOCKET_H
