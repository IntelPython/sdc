#include "utils.hpp"

namespace utils
{

template<>
void* upper_bound(void* first, void* last, void* value, int size, int item_size, void* compare)
{
    void* it = nullptr;
    auto count = size;

    auto less = reinterpret_cast<compare_func>(compare);

    while (count > 0) {
        it = first;
        auto step = count / 2;
        it = advance(it, step, item_size);
        if (!less(value, it)) {
            first = advance(it, 1, item_size);
            count -= step + 1;
        }
        else
            count = step;
    }

    return first;
}

}