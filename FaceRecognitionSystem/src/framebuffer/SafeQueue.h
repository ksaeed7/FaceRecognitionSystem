#ifndef SAFE_QUEUE_H
#define SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class SafeQueue {
public:
    void enqueue(T item);
    T dequeue();
    bool empty() const;
    void signalShutdown();
    size_t max_size = 1000;

private:
    mutable std::mutex m;
    std::queue<T> queue;
    std::condition_variable cv;
    bool shutdown = false;
};

#include "SafeQueue.ipp"

#endif // SAFE_QUEUE_H
