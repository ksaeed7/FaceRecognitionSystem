#include "ServerSocket.h"

// Include additional headers if needed, for example, for logging or processing

ServerSocket::ServerSocket(int port) : port_(port), serverFd_(-1), isRunning_(false) {}

ServerSocket::~ServerSocket() {
    stopServer();
}

bool ServerSocket::startServer(SafeQueue<cv::Mat>& frameQueue) {
    if (isRunning_) return true; // Server is already running

    std::cout<<"Attempting to start server..."<<std::endl;
    serverFd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverFd_ == -1) {
        std::cerr << "Socket creation failed" << std::endl;
        return false;
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(serverFd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        std::cerr << "Socket bind failed" << std::endl;
        close(serverFd_);
        return false;
    }

    if (listen(serverFd_, 10) != 0) {
        std::cerr << "Socket listen failed" << std::endl;
        close(serverFd_);
        return false;
    }

    isRunning_ = true;
    acceptThread_ = std::thread(&ServerSocket::acceptConnections, this, std::ref(frameQueue));
    return true;
}

void ServerSocket::stopServer() {
    if (!isRunning_) return;

    isRunning_ = false;
    close(serverFd_);

    if (acceptThread_.joinable()) {
        acceptThread_.join();
    }
}

void ServerSocket::acceptConnections(SafeQueue<cv::Mat>& frameQueue) {
    std::cout<<"Accepting clients..."<<std::endl;
    while (isRunning_) {
        sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        clientSocket = accept(serverFd_, reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
        std::cout<<clientSocket<<std::endl;
        if (clientSocket == -1) {
            if (isRunning_) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }
        
        //std::cout<< "Client found: "<<clientAddr.sin_addr<<std::endl;
        std::cout<<"Notified Server connect"<<std::endl;
        cv_.notify_one();
        // Handle client connection in a new thread
        std::thread(&ServerSocket::handleClientConnection, this, clientSocket, std::ref(frameQueue)).detach();
    }
}

bool ServerSocket::isClientConnected() {
        // Assuming clientSocket is initialized to -1 and
        // set to a valid socket descriptor upon client connection
        std::cout <<clientSocket<<std::endl;
        if(clientSocket != -1)
        {
            std::cout<<"CLIENT CONNECTION SUCCESSFUL"<<std::endl;
            return true;
        }
        std::cout<<"CLIENT CONNECTION Failed"<<std::endl;
        return false;

    }

void ServerSocket::log(const std::string& message) {
    std::cerr << message << std::endl;
}


void ServerSocket::handleClientConnection(int clientSocket, SafeQueue<cv::Mat>& frameQueue) {
    constexpr size_t MAX_BUFFER_SIZE = 65536; // Adjust as necessary
    try {
        std::cout<<"Handling connection: "<<std::endl;
        while (true) {
            uint32_t imageSize;
            char* headerPtr = reinterpret_cast<char*>(&imageSize);
            int totalHeaderBytesReceived = 0;
            int headerSize = sizeof(imageSize);

            // Receive the header (size of the image data)
            while (totalHeaderBytesReceived < headerSize) {
                int receivedBytes = recv(clientSocket, headerPtr + totalHeaderBytesReceived, headerSize - totalHeaderBytesReceived, 0);
                if (receivedBytes <= 0) {
                    // Handle error or closed connection and exit
                    throw std::runtime_error("Failed to receive header or connection closed.");
                }
                totalHeaderBytesReceived += receivedBytes;
            }

            //std::cout << "Header received. Image size: " << imageSize << std::endl;

            // Allocate buffer for the image and receive the data
            std::vector<unsigned char> buffer(imageSize);
            int totalImageBytesReceived = 0;

            while (totalImageBytesReceived < imageSize) {
                int remainingBytes = imageSize - totalImageBytesReceived;
                int receivedBytes = recv(clientSocket, reinterpret_cast<char*>(buffer.data() + totalImageBytesReceived), remainingBytes, 0);
                if (receivedBytes <= 0) {
                    // Handle error or closed connection
                    throw std::runtime_error("Failed to receive image data or connection closed.");
                }
                totalImageBytesReceived += receivedBytes;
            }

            //std::cout << "Image data received." << std::endl;

            // Decode the received image data
            cv::Mat img = cv::imdecode(cv::Mat(buffer), cv::IMREAD_UNCHANGED);
            if (img.empty()) {
                std::cerr << "Failed to decode image." << std::endl;
                continue; // Proceed to next image if this one failed
            }

            // Process the image as needed (displaying the image is commented out)
            //cv::imshow("Received Image", img);
            //cv::waitKey(1);

            // Enqueue the decoded frame for further processing
            frameQueue.enqueue(img);
        }
        std::cout<<"Exit loop"<<std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error in connection handler: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General exception in connection handler: " << e.what() << std::endl;
    } 
        if (clientSocket != -1) {
            close(clientSocket); // Ensure the socket is closed on function exit
            std::cout << "Socket closed." << std::endl;
        }
        else{
            std::cout<< "Socket closed" <<std::endl;
        }

    
}
