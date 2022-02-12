#pragma once
#pragma once
/* ********************************* FILE ************************************/
/** \file    matlab_helpers.h
 *
 * \brief    basic vector to matlab matrix conversions which account for matlab horribly fixed size array editor window
 *
 * \remark
 * - c++11
 * - no dependencies
 * - fully isolated
 *
 * \todo
 * - Make it write .mat files
 * - rewrite to be more generic
 * - write converters for more general dataforms
 * - use the matlab structs to make the output more generic and still iterable
 * - optionally output a matlab.mat file - unknown compression is used there... 
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2015-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include <utils/cvl/pose.h>

namespace mlib{

template<class T> std::string getMatlabVector(std::vector<T> vs){
    std::stringstream ss;
    ss<<"[";
    for(uint i=0;i<vs.size();++i){
        ss<<vs[i];
        if(i!=(vs.size()-1))
            ss<<"; ";
        if(i>40)
            if((i%50)==0)
            ss<<"\n";
    }
    ss<<"];";
    return ss.str();
}

template<class T>
std::string getMatlabMatrix(std::vector<T> m){
    std::stringstream ss;

    ss<<"[";
    for(uint i=0;i<m.size();++i){
        assert(m[0].size()==m[i].size());
        for(uint j=0;j<m[i].size();++j){// 2 elements
            ss<<m[i][j]<<" ";
        }

        if(i!=(m.size()-1))
            ss<<";";
        if(i%25 ==24)
            ss<<"\n";
    }
    ss<<"];";
    return ss.str();
}


template<class T> std::string getMatlabMatrix(cvl::Pose<T> pose, int precision=6){
    std::stringstream ss;
    cvl::Matrix<T,4,4> M=pose.get4x4();
    ss<<"["<<std::setprecision(precision);
    for(int i=0;i<4;++i){
        for(int j=0;j<4;++j)
            ss<<M(i,j)<<" ";
        ss<<";";
    }
    ss<<"];";
    return ss.str();
}



template<class T> std::string getCSV(std::vector<T> vs){
    std::stringstream ss;        
    for(T v:vs)
      ss<<vs<<", ";
    return ss.str();
}


}// end namespace mlib
