#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "CLog.h"
#include "InstancesManager.h"
#include "pipeline.h"

namespace {
std::atomic_bool running{true};

void signal_handler(int) {
    running = false;
}

std::string config_value(const YAML::Node& root, const char* key, const std::filesystem::path& base) {
    const auto                  value = root[key].as<std::string>();
    const std::filesystem::path path(value);
    return path.is_absolute() ? path.string() : (base / path).lexically_normal().string();
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string base_file        = argc > 1 ? argv[1] : "../config/rk3588/base.yaml";
    int         duration_seconds = argc > 2 ? std::stoi(argv[2]) : 0;
    const auto  base_path        = std::filesystem::absolute(base_file);
    if (!std::filesystem::exists(base_path)) {
        std::cerr << "Base config file not found: " << base_path << '\n';
        return 2;
    }
    const auto base_dir    = base_path.parent_path();
    const auto base_config = YAML::LoadFile(base_path.string());

    LogConfig log_config;
    log_config.out_mode = SYNC;
    log_config.out_pos  = CONSOLE_;
    if (base_config["log_config"] && base_config["log_config"]["log_level"]) {
        log_config.out_level = static_cast<spdlog::level::level_enum>(base_config["log_config"]["log_level"].as<int>());
    }
    if (!Clog::getInstance()->Init(log_config)) {
        return 3;
    }
    if (!base_config["instances"] || !base_config["pipelines"]) {
        LOG_ERROR("RKNN base config requires instances and pipelines");
        return 4;
    }
    if (!InstancesManager::get()->init(config_value(base_config, "instances", base_dir))) {
        LOG_ERROR("RKNN instance initialization failed");
        return 5;
    }
    InstancesManager::get()->start();

    const auto                 pipeline_dir = std::filesystem::path(config_value(base_config, "pipelines", base_dir));
    std::vector<Pipeline::ptr> pipelines;
    for (const auto& entry : std::filesystem::directory_iterator(pipeline_dir)) {
        if (entry.path().extension() != ".yaml") {
            continue;
        }
        const auto config = YAML::LoadFile(entry.path().string());
        if (!config["inputs"]) {
            LOG_WARN("Skipping pipeline without inputs: {}", entry.path().string());
            continue;
        }
        for (size_t index = 0; index < config["inputs"].size(); ++index) {
            auto pipeline = std::make_shared<Pipeline>(entry.path().stem().string() + std::to_string(index));
            if (!pipeline->init(entry.path().string(), static_cast<int>(index))) {
                LOG_ERROR("Failed to initialize pipeline {}", entry.path().string());
                InstancesManager::get()->stop();
                return 6;
            }
            pipeline->start();
            pipelines.push_back(std::move(pipeline));
        }
    }
    if (pipelines.empty()) {
        LOG_ERROR("No RKNN pipeline configs found in {}", pipeline_dir.string());
        InstancesManager::get()->stop();
        return 7;
    }
    if (const char* ready_file = std::getenv("RKNODE_READY_FILE"); ready_file != nullptr && ready_file[0] != '\0') {
        const std::filesystem::path path(ready_file);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream ready(path, std::ios::out | std::ios::trunc);
        if (!ready) {
            LOG_ERROR("Could not write runtime readiness file {}", path.string());
            for (auto& pipeline : pipelines) {
                pipeline->stop();
            }
            InstancesManager::get()->stop();
            return 8;
        }
        ready << "ready\n";
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    const auto deadline = duration_seconds > 0
                              ? std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds)
                              : std::chrono::steady_clock::time_point::max();
    while (running && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    for (auto& pipeline : pipelines) {
        pipeline->stop();
    }
    InstancesManager::get()->stop();
    Clog::getInstance()->UnInit();
    return 0;
}
