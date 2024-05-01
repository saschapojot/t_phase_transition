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
/// @return beta*U
double mc1d::f(const arma::dcolvec& x){

    return this->beta* this->U(x);
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
/// @param x position
/// @return beta * grad U
arma::dcolvec mc1d::gradf(const arma::dcolvec& x){
    return this->beta*this->gradU(x);
}


///
/// @param x x position
/// @return proposed new position
arma::dcolvec mc1d::proposal(const arma::dcolvec& x){



    arma::dcolvec gradVec=this->gradf(x);
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

    arma::dcolvec gradf_z=this->gradf(z);
    arma::dcolvec gradf_x=this->gradf(x);


    double norm_numerator=arma::norm(x-z+h*gradf_z,2);
    double numerator=-f(z)-1.0/(4.0*h)*std::pow(norm_numerator,2);

    double norm_denominator=arma::norm(z-x+h*gradf_x,2);
    double denominator=-f(x)-1.0/(4.0*h)*std::pow(norm_denominator,2);

    double ratio=std::exp(numerator-denominator);

    return std::min(1.0,ratio);



}

///
/// @param filename xml file name of vec
///@param vec vector to be saved
void mc1d::saveVecToXML(const std::string &filename,const std::vector<double> &vec){
    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa & BOOST_SERIALIZATION_NVP(vec);
}


///
/// @param filename  xml file name of vecvec
/// @param vecvec vector<vector> to be saved
void mc1d::saveVecVecToXML(const std::string &filename,const std::vector<std::vector<double>> &vecvec){


    std::ofstream ofs(filename);
    boost::archive::xml_oarchive oa(ofs);
    oa & BOOST_SERIALIZATION_NVP(vecvec);

}

///
/// @param lag decorrelation length
/// @param loopTotal total mc steps
/// @param equilibrium whether equilibrium has reached
/// @return last position
std::vector<double>  mc1d::readEqMc(int& lag,int &loopTotal,bool &equilibrium, bool &same){
    std::random_device rd;
    std::ranlux24_base e2(rd());
    std::uniform_real_distribution<> distUnif01(0, 1);//[0,1)

    //generate initial values of positions
    double leftEnd=-100.0;
    double rightEnd=100.0;
    std::uniform_real_distribution<> distUnifLeftRight(leftEnd,rightEnd);


    arma::dcolvec  xCurr(N);
    for(int j=0;j<N;j++){
        xCurr(j)=distUnifLeftRight(e2);
    }

    //initial value of energy
    double UCurr=this->U(xCurr);
    //output directory
    std::ostringstream sObjT;
    sObjT << std::fixed;
    sObjT << std::setprecision(10);
    sObjT << T;
    std::string TStr = sObjT.str();

    std::ostringstream sObj_a;
    sObj_a<<std::fixed;
    sObj_a<<std::setprecision(10);;
    sObj_a<<a;
    std::string aStr=sObj_a.str();

    std::string outDir="./data/a"+aStr+"/T"+TStr+"/";

    std::string outUAllSubDir=outDir+"UAll/";
    std::string out_xAllSubDir=outDir+"xAll/";

    if (!fs::is_directory(outUAllSubDir) || !fs::exists(outUAllSubDir)) {
        fs::create_directories(outUAllSubDir);
    }

    if (!fs::is_directory(out_xAllSubDir) || !fs::exists(out_xAllSubDir)) {
        fs::create_directories(out_xAllSubDir);
    }
    std::regex stopRegex("stop");
    std::regex wrongRegex("wrong");
    std::regex ErrRegex("Err");
    std::regex lagRegex("lag=\\s*(\\d+)");
    std::regex fileNumRegex("fileNum=\\s*(\\d+)");
    std::regex sameRegex("same");
    std::regex eqRegex("equilibrium");

    std::smatch matchUStop;
    std::smatch matchUWrong;
    std::smatch matchUErr;
    std::smatch matchULag;
    std::smatch matchFileNum;
    std::smatch matchUSame;
    std::smatch matchUEq;

    int counter = 0;
    int fls = 0;
    bool active = true;
    const auto tMCStart{std::chrono::steady_clock::now()};
     std::vector<double> last_x;

    while(fls<this->flushMaxNum and active==true){
        std::vector<std::vector<double>>xAllPerFlush;
        std::vector<double> UAllPerFlush;

        int loopStart=fls*moveNumInOneFlush;
        for(int i=0;i<moveNumInOneFlush;i++){

            //propose a move
            arma::dcolvec xNext= proposal(xCurr);
            double r= acceptanceRatio(xCurr,xNext);
            double u = distUnif01(e2);
            counter++;
            if(u<=r){
                xCurr=xNext;
                UCurr= U(xCurr);
            }//end if

            xAllPerFlush.push_back(arma::conv_to<std::vector<double>>::from(xCurr));
            UAllPerFlush.push_back(UCurr);


        }// end for loop


        int loopEnd = loopStart +moveNumInOneFlush-1;
        std::string filenameMiddle = "loopStart" + std::to_string(loopStart) +
                                     "loopEnd" + std::to_string(loopEnd) + "T" + TStr;

        std::string outUFileName=outUAllSubDir+filenameMiddle+".UAll.xml";
        this->saveVecToXML(outUFileName,UAllPerFlush);

        std::string outxFileName=out_xAllSubDir+filenameMiddle+".xAll.xml";
        this->saveVecVecToXML(outxFileName,xAllPerFlush);
        const auto tflushEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{tflushEnd - tMCStart};
        std::cout << "flush " << fls << std::endl;
        std::cout << "time elapsed: " << elapsed_seconds.count() / 3600.0 << " h" << std::endl;


        //communicate with python to inquire equilibrium

        //inquire equilibrium of U
        std::string commandU = "python3 checkVec.py " + outUAllSubDir;
        std::string resultU;
        if (fls % 6 == 5) {
            try {
                resultU = this->execPython(commandU.c_str());
                std::cout << "U message from python: " << resultU << std::endl;

            }
            catch (const std::exception &e) {
                std::cerr << "Error: " << e.what() << std::endl;
                std::exit(10);
            }
            catch (...) {
                // Handle any other exceptions
                std::cerr << "Error" << std::endl;
                std::exit(11);
            }
            // parse result
            if (std::regex_search(resultU, matchUErr, ErrRegex)) {
                std::cout << "error encountered" << std::endl;
                std::exit(12);
            }

            if (std::regex_search(resultU, matchUWrong, wrongRegex)) {
                std::exit(13);
            }

            if (std::regex_search(resultU, matchUStop, stopRegex)){
                if (std::regex_search(resultU, matchUSame, sameRegex)){
                    active = false;
                    same=true;
                    std::regex_search(resultU, matchFileNum, fileNumRegex);
                    std::string fileNumStr = matchFileNum.str(1);
                    this->lastFileNum = std::stoi(fileNumStr);
                    last_x=xAllPerFlush[xAllPerFlush.size()-1];


                }


            }//end of regex search

            if (std::regex_search(resultU, matchUEq, eqRegex)){
                if (std::regex_search(resultU, matchULag, lagRegex)){

                    std::string lagStrU = matchULag.str(1);
                    int lagU = std::stoi(lagStrU);
                    std::cout << "lag=" << lagU << std::endl;
                    std::regex_search(resultU, matchFileNum, fileNumRegex);
                    std::string fileNumStr = matchFileNum.str(1);
                    this->lastFileNum = std::stoi(fileNumStr);
                    active = false;
                    last_x=xAllPerFlush[xAllPerFlush.size()-1];
                }


            }//end of regex search






        }//end if

        fls++;



    }//end while

    loopTotal=counter;

    std::ofstream  outSummary(outDir+"summary.txt");
    const auto tMCEnd{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
    outSummary << "total mc time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    outSummary << "total loop number: " << loopTotal << std::endl;

    outSummary << "lastFileNum=" << lastFileNum << std::endl;
    outSummary << "equilibrium reached: " << !active << std::endl;
    outSummary << "same: " << same << std::endl;

    outSummary << "lag=" << lag << std::endl;
    outSummary.close();

    return last_x;



}//end of function



///
/// @param lag decorrelation length
/// @param loopEq total loop numbers in reaching equilibrium
///@param x_init x from readEqMc
void mc1d::executionMCAfterEq(const int& lag,const int & loopEq, const std::vector<double>& x_init){
    int counter=0;
    int remainingDataNum = this->dataNumTotal-lastFileNum*moveNumInOneFlush;

    int remainingLoopNum = remainingDataNum * lag;
    if (remainingLoopNum <= 0) {
        return;
    }

    double remainingLoopNumDB = static_cast<double>(remainingLoopNum);
    double remainingFlushNumDB = std::ceil(remainingLoopNumDB/moveNumInOneFlush);
    int remainingFlushNum = static_cast<int>(remainingFlushNumDB);

    std::random_device rd;
    std::ranlux24_base e2(rd());
    std::uniform_real_distribution<> distUnif01(0, 1);//[0,1)

    arma::dcolvec xCurr(x_init);
    double UCurr=this->U(xCurr);

    //output directory
    std::ostringstream sObjT;
    sObjT << std::fixed;
    sObjT << std::setprecision(10);
    sObjT << T;
    std::string TStr = sObjT.str();

    std::ostringstream sObj_a;
    sObj_a<<std::fixed;
    sObj_a<<std::setprecision(10);;
    sObj_a<<a;
    std::string aStr=sObj_a.str();

    std::string outDir="./data/a"+aStr+"/T"+TStr+"/";

    std::string outUAllSubDir=outDir+"UAll/";
    std::string out_xAllSubDir=outDir+"xAll/";

    const auto tMCStart{std::chrono::steady_clock::now()};

    std::cout<<"remaining flush number: "<<remainingFlushNum<<std::endl;

    for (int fls = 0; fls < remainingFlushNum; fls++) {

        std::vector<std::vector<double>>xAllPerFlush;
        std::vector<double> UAllPerFlush;
        int loopStart =loopEq+fls*moveNumInOneFlush;
        for(int i=0;i<moveNumInOneFlush;i++){
            //propose a move
            arma::dcolvec xNext= proposal(xCurr);
            double r= acceptanceRatio(xCurr,xNext);
            double u = distUnif01(e2);
            counter++;
            if(u<=r){
                xCurr=xNext;
                UCurr= U(xCurr);
            }//end if
            xAllPerFlush.push_back(arma::conv_to<std::vector<double>>::from(xCurr));
            UAllPerFlush.push_back(UCurr);


        }//end for loop

        int loopEnd = loopStart +moveNumInOneFlush-1;
        std::string filenameMiddle = "loopStart" + std::to_string(loopStart) +
                                     "loopEnd" + std::to_string(loopEnd) + "T" + TStr;
        std::string outUFileName=outUAllSubDir+filenameMiddle+".UAll.xml";
        this->saveVecToXML(outUFileName,UAllPerFlush);

        std::string outxFileName=out_xAllSubDir+filenameMiddle+".xAll.xml";
        this->saveVecVecToXML(outxFileName,xAllPerFlush);
        const auto tflushEnd{std::chrono::steady_clock::now()};
        const std::chrono::duration<double> elapsed_seconds{tflushEnd - tMCStart};
        std::cout << "flush " << fls << std::endl;
        std::cout << "time elapsed: " << elapsed_seconds.count() / 3600.0 << " h" << std::endl;


    }//end of flush loop

    std::ofstream outSummary(outDir + "summaryAfterEq.txt");
    int loopTotal=counter;
    const auto tMCEnd{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{tMCEnd - tMCStart};
    outSummary << "total mc time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    outSummary << "total loop number: " << loopTotal << std::endl;



}//end of mc