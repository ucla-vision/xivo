#include "tumvi.h"
#include "viewer.h"
#include "gtest/gtest.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

using namespace feh;

static std::string tumvi_root = "/local2/Data/tumvi/exported/euroc/512_16/";

class LoaderTest: public ::testing::Test {
};

TEST_F(LoaderTest, Init) 
{
    std::string dataset = tumvi_root + "/dataset-room1_512_16/mav0/";
    TUMVILoader loader(dataset+"/cam0/", dataset+"/imu0/");
    ASSERT_GT(loader.size(), 0) << "Wrong dataset path?";
    std::cout << "#entries=" << loader.size() << std::endl;
    Viewer viewer;
    viewer.Initialize("cfg/viewer.json");

    for (int i = 0; i < loader.size(); ++i) {
        auto entry = loader.Get(i);
        std::cout << entry;
        if (entry.type_ == msg::Generic::IMAGE) {
            auto image = cv::imread(entry.image_path_);
            viewer.Update(image);
            viewer.Refresh();
        }
    }

}

