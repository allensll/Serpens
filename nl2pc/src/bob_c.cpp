#include "nl2pc_ckks.h"


int main() {

    NLParty& party = create(ckksCLIENT, "127.0.0.1", 14714, 8, false, true);

    size_t n = 802816;
    std::vector<double> x2(n);
    for (size_t i=0; i<n; i++) {
        x2[i] = double(rand() % 1000000) / 1000000 - 1;
    }
    // std::vector<double> x2{-1,1,1,3};

    for (size_t i=0; i<5; i++) {
        // std::vector<double> res = party.cmp(x2);

        // std::vector<double> res = party.relu(x2, false);

        std::vector<double> res = party.maxpool2d(x2, 4, false);

        // std::cout << res[0] << " || " << res[1] << " || " << res[2] << " || " << res[3] << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // delete party;

    return 0;
}