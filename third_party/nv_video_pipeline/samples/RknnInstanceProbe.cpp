#include <yaml-cpp/yaml.h>

#include <iostream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <utility>

#include "BaseData.h"
#include "CLog.h"
#include "FrameMeta.h"
#include "InstancesManager.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: rknn_instance_probe <instances.yaml> <instance> [instance...]\n";
        return 2;
    }

    LogConfig log_config;
    log_config.out_mode  = SYNC;
    log_config.out_pos   = CONSOLE_;
    log_config.out_level = spdlog::level::info;
    if (!Clog::getInstance()->Init(log_config)) {
        return 3;
    }
    if (!InstancesManager::get()->init(argv[1])) {
        LOG_ERROR("RKNN probe could not initialize {}", argv[1]);
        Clog::getInstance()->UnInit();
        return 4;
    }
    InstancesManager::get()->start();

    int status = 0;
    for (int index = 2; index < argc; ++index) {
        const std::string name(argv[index]);
        if (!InstancesManager::get()->has_key(name)) {
            LOG_ERROR("RKNN probe instance {} does not exist", name);
            status = 5;
            break;
        }
        auto data = std::make_shared<Data::BaseData>();
        data->set_frame_meta(std::make_shared<object_meta::FrameMeta>(cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 0, 0))));
        Job job;
        job.data    = std::move(data);
        auto future = job.promise->get_future();
        if (!InstancesManager::get()->get_instance(name)->commit(job) || !future_wait_for_true(future, 30000)) {
            LOG_ERROR("RKNN probe inference failed for {}", name);
            status = 6;
            break;
        }
        LOG_INFO("RKNN probe inference succeeded for {}", name);
    }

    InstancesManager::get()->stop();
    Clog::getInstance()->UnInit();
    return status;
}
