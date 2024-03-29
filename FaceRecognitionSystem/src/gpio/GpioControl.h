#ifndef GPIOCONTROL_H
#define GPIOCONTROL_H

#include <string> // Make sure to include necessary headers
#include <fstream> // For std::ofstream and std::ifstream

class GpioControl {
public:
    GpioControl(int pin);
    ~GpioControl();

    void exportPin();
    void unexportPin();
    void setDirection(std::string dir);
    void writePin(int value);
    int readPin();

private:
    int pin;
    std::string basePath;
    std::string pinPath;
};

#endif // GPIOCONTROL_H
