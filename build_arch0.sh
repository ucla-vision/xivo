CPU_COUNT=4
PROJECT_DIR=$(pwd)
echo $PROJECT_DIR

cd $PROJECT_DIR/thirdparty/gflags
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install -j $CPU_COUNT

cd $PROJECT_DIR/thirdparty/glog
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. 
make install -j $CPU_COUNT

