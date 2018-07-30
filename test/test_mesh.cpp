#include <iostream>

#include "../src/mesh.hpp"

int main() {
    Mesh mesh;
    mesh.init(Size(5,6));

    std::cout << mesh.Q() << std::endl;

    return 0;
}