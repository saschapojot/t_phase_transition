//
// Created by polya on 4/30/24.
//

#ifndef T_PHASE_TRANSITION_1D_HPP
#define T_PHASE_TRANSITION_1D_HPP

#include <algorithm>
#include <armadillo>
#include <array>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = boost::filesystem;
//this subroutine computes the mc evolution for a 1d system
class mc1d {
public:mc1d(double temperature,double stepSize,double aVal,bool isDiag,int pntNum){
        this->T=temperature;
        this->beta=1/T;
        this->h=stepSize;
        this->a=aVal;
        this->diag=isDiag;
        this->N=pntNum;
        this->eqPositions=arma::dcolvec (N);
        for(int i=0;i<N;i++){
            this->eqPositions(i)=static_cast<double >(i);
        }
//        std::cout<<eqPositions<<std::endl;

    }

public:
    ///
    /// @param x position
    /// @return total potential
    double U(const arma::dcolvec& x);

    ///
    /// @param x position
    /// @return gradient of U with respect to x
    arma::dcolvec gradU(const arma::dcolvec& x);

    ///
    /// @param x position
    /// @return beta*U
    double f(const arma::dcolvec& x);

    ///
    /// @param x position
    /// @return beta * grad U
    arma::dcolvec gradf(const arma::dcolvec& x);

    ///
    /// @param x x position
    /// @return proposed new position
    arma::dcolvec proposal(const arma::dcolvec& x);


    ///
    /// @param cmd python execution string
    /// @return signal from the python
    static std::string execPython(const char* cmd);

    ///
    /// @param x current position
    /// @param z proposed position
    /// @return acceptance ratio
    double acceptanceRatio(const arma::dcolvec& x,const arma::dcolvec& z);

    ///
    /// @param filename xml file name of vec
    ///@param vec vector to be saved
    static  void saveVecToXML(const std::string &filename,const std::vector<double> &vec);


    ///
    /// @param lag decorrelation length
    /// @param loopTotal total mc steps
    /// @param equilibrium whether equilibrium has reached
    void readEqMc(int& lag,int &loopTotal,bool &equilibrium, bool &same);


    ///
    /// @param filename  xml file name of vecvec
    /// @param vecvec vector<vector> to be saved
    static void saveVecVecToXML(const std::string &filename,const std::vector<std::vector<double>> &vecvec);

    ///
    /// @param lag decorrelation length
    /// @param loopEq total loop numbers in reaching equilibrium
    void executionMCAfterEq(const int& lag,const int & loopEq);// mc simulation without inquiring equilibrium after reaching equilibrium


public:
    double T;// temperature
    double beta;
    int moveNumInOneFlush=3000;// flush the results to python every moveNumInOneFlush iterations
    int flushMaxNum=700;
    int dataNumTotal=8000;
    double h;// step size
    double a=5.3;//stiffness
    bool diag=true;// whether the quadratic form of energy is diagonal
    int N=10;//number of points
    arma::dcolvec  eqPositions;// equilibrium positions
    double lastFileNum=0;
};






#endif //T_PHASE_TRANSITION_1D_HPP
