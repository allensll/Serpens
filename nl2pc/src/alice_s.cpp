#include "nl2pc_ckks.h"


int main() {

    NLParty& party = create(ckksSERVER, "127.0.0.1", 14714, 8, false, true);

    size_t n = 802816;
    std::vector<double> x1(n);
    for (size_t i=0; i<n; i++) {
        x1[i] = double(rand() % 1000000) / 1000000;
    }
    // std::vector<double> x1{2,-1.01,-2, 0};

    std::chrono::high_resolution_clock::time_point time_start, time_end;
    std::chrono::microseconds time_diff;

    for (size_t i=0; i<5; i++) {
        time_start = std::chrono::high_resolution_clock::now();

        // std::vector<double> res = party.cmp(x1);

        // std::vector<double> res = party.relu(x1, false);

        std::vector<double> res = party.maxpool2d(x1, 4, false);

        time_end = std::chrono::high_resolution_clock::now();
        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);

        // std::cout << res[0] << " || " << res[1] << " || " << res[2] << " || " << res[3] << std::endl;
        std::cout << "Done [" << time_diff.count() / 1000000.0 << " seconds]" << std::endl;
    }

    // delete party;

    return 0;
}