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
#include <cxxabi.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace fs = boost::filesystem;
//this subroutine computes the mc evolution for a 1d system

class potentialFunction{
public:
    virtual double operator() (const arma::dcolvec& x, const arma::dcolvec & eqPositions)const=0;
    virtual arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions) const=0;
    virtual ~ potentialFunction(){};
public:
    double a=1;//stiffness
};

class quadraticDiag: public  potentialFunction{
    ///
    /// @param x position
    /// @param eqPositions centers
    /// @param a coefficient
    /// @return
    double operator()(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
        arma::dcolvec diff=x-eqPositions;
        double norm2Tmp=arma::norm(diff,2);
        return a*std::pow(norm2Tmp,2);
    }
    arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{

        arma::dcolvec diff=x-eqPositions;
        return 2*a*diff;
    }


};

class quadraticCubicAbs:public potentialFunction{

     double operator() (const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
          arma::dcolvec diff=x-eqPositions;

        double val=0;
        int len=x.size();
        for (int k=0;k<len;k++){
            val+=std::pow(std::abs(diff(k)),3)+std::pow(diff(k),2);
        }
        return a*val;
     }

      arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
        arma::dcolvec diff=x-eqPositions;
        arma::dcolvec gradVec(x.size());
        for(int k=0;k<x.size();k++){
            gradVec(k)=3*std::pow(diff(k),2);

        }
        return a*gradVec%arma::sign(diff)+a*2*diff;
    }

};

class quarticCubicDiag:public potentialFunction{
    double operator() (const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{

        arma::dcolvec diff=x-eqPositions;

        double val=0;
        int len=x.size();
        for (int k=0;k<len;k++){

            val+=10*std::pow(diff(k),4)+std::pow(diff(k),3);
        }

        return a*val;


    }
    arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
        arma::dcolvec diff=x-eqPositions;
        arma::dcolvec gradVec(x.size());
        for(int k=0;k<x.size();k++){
            gradVec(k)=40*std::pow(diff(k),3)+3*std::pow(diff(k),2);
        }
        return a*gradVec;
    }
};

class pdQuadratic: public potentialFunction{
public:
    pdQuadratic(const int &pntNum) {
        //initialize A matrix
        A = arma::dmat(pntNum, pntNum);
        std::seed_seq seq{17, 19, 23};  // Fixed sequence for reproducibility
        std::ranlux24_base gen;
        gen.seed(seq);
        std::uniform_real_distribution<> distr(-1, 1);
        for (int i = 0; i < pntNum; i++) {
            for (int j = 0; j < pntNum; j++) {
                A(i, j) = distr(gen);
            }
        }
        //A to orthogonal
        arma::dmat B=arma::orth(A);
//        std::cout<<"BTB="<<B.t()*B<<std::endl;
        //initialize diagonal matrix
        std::vector<double> diagElems;
        for (int j = 0; j < pntNum; j++) {
            diagElems.push_back(static_cast<double>(j + 1));
        }
        //compute pd matrix
        diag = arma::diagmat(arma::conv_to<arma::colvec>::from(diagElems));
//        std::cout<<diag<<std::endl;
        pdmat = B.t() * diag * B;
//        std::cout<<"det(A)="<<arma::det(A)<<std::endl;
        std::cout<<"det(pdmat)="<<arma::det(pdmat)<<std::endl;

    }//end of constructor

    double operator() (const arma::dcolvec& x, const arma::dcolvec & eqPositions) const override{

    arma::dcolvec diff=x-eqPositions;
    double val=arma::dot(diff,0.5*pdmat*diff);
        return val;
    }

    arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{

        arma::dcolvec diff=x-eqPositions;
        return pdmat*diff;
    }


    arma::dmat A;
    arma::dmat diag;
    arma::dmat pdmat;
//    std::string prefix="ortho";

};


class quarticCubicQudraticDiag: public potentialFunction{
    double operator() (const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
        arma::dcolvec diff=x-eqPositions;
//        double val=0;
//        for(int k=0;k<x.size();k++){
//            double diffk=diff(k);
//            val+=10*std::pow(diffk,4)+std::pow(diffk,3)+3.5*std::pow(diffk,2);
//        }
//        return a*val;
        arma::dcolvec vec=arma::pow(diff,2)+arma::abs(arma::pow(diff,3))+arma::pow(diff,4);
        return a*arma::sum(vec);

    }

    arma::dcolvec grad(const arma::dcolvec& x, const arma::dcolvec & eqPositions)const override{
        arma::dcolvec diff=x-eqPositions;

arma::dcolvec  gradVec=2*diff+3*arma::pow(diff,2)%arma::sign(diff)+4*arma::pow(diff,3);
        return a*gradVec;

    }

};

class mc1d {
public:mc1d(double temperature,double stepSize,int pntNum, const std::shared_ptr<potentialFunction>& funcPtr,int withGrd ){
        this->T=temperature;
        this->beta=1/T;
        this->h=stepSize;
        this->potFuncPtr=funcPtr;

//        this->diag=isDiag;
        this->N=pntNum;
        this->withGrad=withGrd;
        this->eqPositions=arma::dcolvec (N);
        for(int i=0;i<N;i++){
            this->eqPositions(i)=static_cast<double >(i);
        }
        std::cout<<"withGrad="<<this->withGrad<<std::endl;
//        std::cout<<eqPositions<<std::endl;

    }

public:
    ///
    /// @param x position
    /// @return total potential
//    double U(const arma::dcolvec& x);

    ///
    /// @param x position
    /// @return gradient of U with respect to x
//    arma::dcolvec gradU(const arma::dcolvec& x);

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
    /// @return last position
    std::vector<double> readEqMc(int& lag,int &loopTotal,bool &equilibrium, bool &same);


    ///
    /// @param filename  xml file name of vecvec
    /// @param vecvec vector<vector> to be saved
    static void saveVecVecToXML(const std::string &filename,const std::vector<std::vector<double>> &vecvec);

    ///
    /// @param lag decorrelation length
    /// @param loopEq total loop numbers in reaching equilibrium
    ///@param x_init x from readEqMc
    void executionMCAfterEq(const int& lag,const int & loopEq, const std::vector<double>& x_init);// mc simulation without inquiring equilibrium after reaching equilibrium


    std::string demangle(const char* name) {
        int status = -1;
        char* demangled = abi::__cxa_demangle(name, NULL, NULL, &status);
        std::string result(name);
        if (status == 0) {
            result = demangled;
        }
        std::free(demangled);
        return result;
    }

public:
    double T;// temperature
    double beta;
    int moveNumInOneFlush=3000;// flush the results to python every moveNumInOneFlush iterations
    int flushMaxNum=700;
    int dataNumTotal=8000;
    double h;// step size
//    double a=5.3;//stiffness
//    bool diag=true;// whether the quadratic form of energy is diagonal
    int N=10;//number of points
    arma::dcolvec  eqPositions;// equilibrium positions
    double lastFileNum=0;
    std::shared_ptr<potentialFunction> potFuncPtr;
    int withGrad=1;

};






#endif //T_PHASE_TRANSITION_1D_HPP
