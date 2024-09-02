#include <iostream>
#include <vector>

void push(std::vector<const char*> & container) { const char* a = "abc";
    container.push_back(a);
}

int main()
{
    std::vector<const char*> container_;
    push(container_);
    auto a_ = container_.data();
    std::cout << *a_ << std::endl;
}