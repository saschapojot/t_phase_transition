//
// Created by polya on 4/30/24.
//

#include "1d.hpp"
///
/// @param x position
/// @return total potential
double mc1d::U(const arma::dcolvec& x){

    arma::dcolvec diff=x-this->eqPositions;
    double norm2Tmp=arma::norm(diff,2);
    return this->a*std::pow(norm2Tmp,2);

}



///
/// @param x position
/// @return gradient of U with respect to x
arma::dcolvec mc1d::gradU(const arma::dcolvec& x){
    arma::dcolvec  grad(N);
    for(int j=0;j<N;j++){
        grad(j)=2*a*(x(j)-static_cast<double >(j));
    }

    return grad;


}


///
/// @param x x position
/// @return proposed new position
arma::dcolvec mc1d::proposal(const arma::dcolvec& x){



    arma::dcolvec gradVec=this->gradU(x);
    arma::dcolvec meanVec=x-h*gradVec;
    double stddev=std::sqrt(2.0*h);

    std::random_device rd;
    std::ranlux24_base gen(rd());

    arma::dcolvec z(N);//proposed new position

    for(int j=0;j<N;j++){
        std::normal_distribution<double>  dTmp(meanVec(j),stddev);
        z(j)=dTmp(gen);
    }

    return z;
}


///
/// @param cmd python execution string
/// @return signal from the python
std::string mc1d::execPython(const char* cmd){

    std::array<char, 4096> buffer; // Buffer to store command output
    std::string result; // String to accumulate output

    // Open a pipe to read the output of the executed command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    // Read the output a chunk at a time and append it to the result string
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result; // Return the accumulated output


}


///
/// @param x current position
/// @param z proposed position
/// @return acceptance ratio
double mc1d::acceptanceRatio(const arma::dcolvec& x,const arma::dcolvec& z){

    arma::dcolvec gradU_z=this->gradU(z);
    arma::dcolvec gradU_x=this->gradU(x);


    double norm_numerator=arma::norm(x-z+h*gradU_z,2);
    double numerator=-U(z)-1.0/(4.0*h)*std::pow(norm_numerator,2);

    double norm_denominator=arma::norm(z-x+h*gradU_x,2);
    double denominator=-U(x)-1.0/(4.0*h)*std::pow(norm_denominator,2);

    double ratio=std::exp(numerator-denominator);

    return std::min(1.0,ratio);



}

///
/// @param filename xml file name of vec
///@param vec vector to be saved
void saveVecToXML(const std::string &filename,const std::vector<double> &vec){
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa & BOOST_SERIALIZATION_NVP(vec);
}

///
/// @param lag decorrelation length
/// @param loopTotal total mc steps
/// @param equilibrium whether equilibrium has reached
void mc1d::readEqMc(int& lag,int &loopTotal,bool &equilibrium){
    std::random_device rd
    std::ranlux24_base e2(rd());
    std::uniform_real_distribution<> distUnif01(0, 1);//[0,1)


}