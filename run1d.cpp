//
// Created by polya on 4/30/24.
//

#include "./1d/1d.hpp"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    double T = std::stod(argv[1]);
    double a = std::stod(argv[2]);
    double stepSize = 0.01;
    bool diag = true;
    int dataNum = 10000;
    auto mc1dObj = mc1d(T, stepSize, a, diag, dataNum);
    int lag=-1;
    int totalLoopEq=0;
    bool eq=false;
    bool same= false;

   std::vector<double>last_x= mc1dObj.readEqMc(lag,totalLoopEq,eq,same);
    if(!same and lag>0){
        mc1dObj.executionMCAfterEq(lag,totalLoopEq,last_x);
    }

    return 0;
}