#include "GpioControl.h"
#include <unistd.h>\
#include <fstream> // For std::ofstream and std::ifstream
#include <string>  // For std::string

GpioControl::GpioControl(int pin) : pin(pin), basePath("/sys/class/gpio/"), pinPath(basePath + "gpio" + std::to_string(pin) + "/") {
    exportPin();
    usleep(100000); // Wait for the GPIO files to become available in sysfs
}

GpioControl::~GpioControl() {
    unexportPin();
}

void GpioControl::exportPin() {
    std::ofstream file(basePath + "export");
    if (file.is_open()) {
        file << pin;
        file.close();
    }
}

void GpioControl::unexportPin() {
    std::ofstream file(basePath + "unexport");
    if (file.is_open()) {
        file << pin;
        file.close();
    }
}

void GpioControl::setDirection(std::string dir) {
    std::ofstream file(pinPath + "direction");
    if (file.is_open()) {
        file << dir;
        file.close();
    }
}

void GpioControl::writePin(int value) {
    std::ofstream file(pinPath + "value");
    if (file.is_open()) {
        file << value;
        file.close();
    }
}

int GpioControl::readPin() {
    std::ifstream file(pinPath + "value");
    int value = -1;
    if (file.is_open()) {
        file >> value;
        file.close();
    }
    return value;
}
