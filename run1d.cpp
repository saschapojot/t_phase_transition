//
// Created by polya on 4/30/24.
//

#include "/home/polya/Documents/cppCode/t_phase_transition/1d/1d.hpp"

int main(int argc, char *argv[]) {

mc1d obj1d(3,0.01,4,true,10);
    int lag=0;
    int loopTotal=0;
    bool equilibrium;

arma::dcolvec x{1,2,3};
std::vector<double> y=arma::conv_to<std::vector<double>>::from(x);
for(const auto&val: y){
    std::cout<<val<<",";
}
std::cout<<std::endl;
}