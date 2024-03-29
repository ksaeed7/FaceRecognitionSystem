// This file should be included at the end of the SafeQueue.h file
// or in the SafeQueue.h file directly after the class definition.

template<typename T>
void SafeQueue<T>::enqueue(T item) {
    std::lock_guard<std::mutex> lock(m);
    if(shutdown) return;
    if(queue.size() >= max_size) {
        std::queue<T> empty;
        std::swap(queue, empty); // Instantly clears the queue
        // Optionally, log this event or handle it appropriately
    }
    queue.push(std::move(item));
    cv.notify_one();
}


template<typename T>
T SafeQueue<T>::dequeue() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [this] { return !queue.empty() || shutdown; });
    if(shutdown && queue.empty()) throw std::runtime_error("Shutdown and empty queue");
    T item = std::move(queue.front());
    queue.pop();
    return item;
}

template<typename T>
bool SafeQueue<T>::empty() const {
    std::lock_guard<std::mutex> lock(m);
    return queue.empty();
}

template<typename T>
void SafeQueue<T>::signalShutdown() {
    std::lock_guard<std::mutex> lock(m);
    shutdown = true;
    cv.notify_all();
}
