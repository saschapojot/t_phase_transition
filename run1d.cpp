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
    int which=std::stoi(argv[2]);//which potential function to use

    double stepSize = 0.01;
    std::vector<std::shared_ptr<potentialFunction>> UFuncs;


    int dataNum = 10;
    UFuncs.push_back(std::make_shared<quadraticDiag>());//0th function: quadratic
    UFuncs.push_back(std::make_shared<quarticCubicDiag>());//1st function: quartic+cubic
    UFuncs.push_back(std::make_shared<pdQuadratic>(dataNum));//2nd function: quadratic form
    UFuncs.push_back(std::make_shared<quadraticCubicQudraticDiag>());//3rd function: quartic+cubic+quadratic

    auto mc1dObj = mc1d(T, stepSize, dataNum,UFuncs[which]);
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