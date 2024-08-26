#include <iostream>
#include <string>

int main() {
    long long int i = (0);
    long long int count = (0);
    long long int temp = (0);
    long long int prev = (0);
    long long int current = (1);
    std::cout << "How many fibonacci numbers to generated" << '\n';
    std::cin >> count;
    std::cout << "" << '\n';
    std::cout << prev << '\n';
    std::cout << current << '\n';
    for (; ((i) < (count)); i += (1)) {
        temp = (current);
        current += (prev);
        prev = (temp);
        std::cout << current << '\n';
    }

    return 0;
}