#!/bin/sh
PROJECT_DIR=$(pwd)
echo $PROJECT_DIR

cd $PROJECT_DIR/thirdparty/googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/gflags
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/glog
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/eigen3
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/Sophus
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/Pangolin
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j

cd $PROJECT_DIR/thirdparty/jsoncpp
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=TRUE
make install -j

# to build gperftools, need to install autoconf and libtool first
# sudo apt-get install autoconf libtool
cd $PROJECT_DIR/thirdparty/gperftools
./autogen.sh
./configure --prefix=$PROJECT_DIR/thirdparty/gperftools
make install


mkdir ${PROJECT_DIR}/build
cd ${PROJECT_DIR}/build
cmake ..
make -j
