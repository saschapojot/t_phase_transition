//
// Created by polya on 4/30/24.
//

#include "/home/polya/Documents/cppCode/t_phase_transition/1d/1d.hpp"

int main(int argc, char *argv[]) {

mc1d obj1d(3,0.01,4,true,10);
arma::dcolvec x{9,8,7,6,5,4,3,2,1,0};
std::cout<<obj1d.proposal(x)<<std::endl;
}