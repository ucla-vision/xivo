#!/bin/bash

# parsing options
BUILD_ROSNODE=false
BUILD_G2O=false

for arg in "$@"
do
  if [ "$arg" == "ros" ]; then
    BUILD_ROSNODE=true
  fi

  if [ "$arg" == "g2o" ]; then
    BUILD_G2O=true
  fi
done

if [ "$BUILD_ROSNODE" = true ]; then
  echo "BUILD WITH ROS SUPPORT"
fi

if [ "$BUILD_G2O" = true ]; then
  echo "BUILD WITH G2O SUPPORT"
fi


# build dependencies
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

cd $PROJECT_DIR/thirdparty/eigen-3.3.7
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
# cd $PROJECT_DIR/thirdparty/gperftools
# ./autogen.sh
# ./configure --prefix=$PROJECT_DIR/thirdparty/gperftools
# make install

if [ "$BUILD_G2O" = true ]; then
  cd $PROJECT_DIR/thirdparty/g2o
  mkdir build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=../release
  make install -j
fi


# build xivo
mkdir ${PROJECT_DIR}/build
cd ${PROJECT_DIR}/build

if [ "$BUILD_G2O" = true ] && [ "$BUILD_ROSNODE" = true ]; then
  cmake .. -DBUILD_G2O=TRUE -DBUILD_ROSNODE=TRUE
elif [ "$BUILD_G2O" = true ]; then
  cmake .. -DBUILD_G2O=TRUE
elif [ "$BUILD_ROSNODE" = true ]; then
  cmake .. -DBUILD_ROSNODE=TRUE
else
  cmake ..
fi
make -j
