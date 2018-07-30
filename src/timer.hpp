#pragma once

#include <iostream>
#include <ctime>

using std::cout;
using std::endl;

class Timer {
private:
    std::clock_t start;

public:
    void reset() {
        start = std::clock();
    }

    void tick(const std::string& msg) {
        double step = double(std::clock() - start) * 1000.0 / CLOCKS_PER_SEC;
        cout << "[time used: " << step << " ms] - " << msg << endl;
        reset();
    }
};