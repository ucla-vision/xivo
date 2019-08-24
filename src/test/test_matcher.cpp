#include "gtest/gtest.h"

#include "opencv2/highgui/highgui.hpp"

#include "visualize.h"
#include "matcher.h"
#include "tumvi.h"

using namespace xivo;

static std::string dataroot = "/home/feixh/Data/tumvi/exported/euroc/512_16/";

class MatcherTest: public ::testing::Test {
};

TEST_F(MatcherTest, Init)
{
    Matcher matcher("../cfg/tracker.json");
}

TEST_F(MatcherTest, Detect)
{
    Matcher matcher("../cfg/tracker.json");

    std::string dataset = dataroot + "/dataset-corridor1_512_16/mav0/";
    TUMVILoader loader(dataset+"/cam0/", dataset+"/imu0/");
    for (int i = 0; i < loader.size(); ++i) {
        auto entry = loader.Get(i);
        if (entry.type_ == msg::Generic::IMAGE) {
            std::cout << entry.image_path_ << std::endl;
            auto image = cv::imread(entry.image_path_);
            cv::imshow("image", image);

            // overwrite
            matcher.initialized_ = false;
            matcher.features_.clear();

            matcher.Update(image);

            auto disp = matcher.VisualizeTracks();
            cv::imshow("detect", disp);

            char ckey = cv::waitKey(0);
            if (ckey == 'q') break;
        }
    }
}

