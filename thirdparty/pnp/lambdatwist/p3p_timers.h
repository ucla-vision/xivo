#pragma once
#if P3P_WITH_TIMING

#include "mlib/utils/mlibtime.h"
mlib::Timer lt1("lambda twist timer 1: ");
mlib::Timer lt2("lambda twist timer 2: ");
mlib::Timer lt3("lambda twist timer 3: ");
mlib::Timer lt4("lambda twist timer 4: ");
mlib::Timer lt5("lambda twist timer 5: ");
mlib::Timer lt6("lambda twist timer 6: ");
mlib::Timer lt7("lambda twist timer 7: ");
mlib::Timer lt8("lambda twist timer 8: ");
mlib::Timer lt9("lambda twist timer 9: ");

mlib::Timer teig1("eigenvalue 1 timer"),teig2("eigenvalue 2 timer");
mlib::Timer t1,t2,t3,t4,t5,t6,t7,t8,t9,ttot;

/**
 * @brief printtimers displays the internal timing information for the solver -DP3P_WITH_TIMING is defined
 */
inline void printtimers(){


    std::vector<mlib::Timer> ts={t1,t2,t3,t4,t5,t6,t7,t8,teig1,teig2};
    std::cout<<ts<<std::endl;

    std::vector<mlib::Timer> ts2={lt1,lt2,lt3,lt4,lt5,lt6,lt7,lt8,lt9};
    std::cout<<ts2<<std::endl;

}


#define TIC(timer) timer.tic()
#define TOC(timer) timer.toc()
#else
void printtimers(){}
#define TIC(timer)
#define TOC(timer)
#endif

