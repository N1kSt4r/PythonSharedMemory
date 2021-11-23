#include <iostream>

#include "SharedStore.h"

int main() {
    std::cout << "hello world";
    SharedStore<float> store("check", 40, true);

    return 0;
}
