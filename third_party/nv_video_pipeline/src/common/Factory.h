#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "CLog.h"

template <typename T, typename... Args>
class Factory {
public:
    static Factory& Instance() {
        if (!instance_) {
            instance_ = new Factory<T, Args...>();
        }
        return *instance_;
    }

    void Register(const std::string& name, std::function<std::shared_ptr<T>(Args...)> creator) {
        creators_[name] = creator;
    }

    std::shared_ptr<T> Create(const std::string& name, Args... args) {
        if (creators_.find(name) == creators_.end()) {
            LOG_ERROR("Factory: {} not registered", name);
            exit(EXIT_FAILURE);
        }

        return creators_.find(name) == creators_.end() ? std::shared_ptr<T>() : creators_[name](args...);
    }

private:
    Factory() {
    }

    static Factory<T, Args...>*                                                 instance_;
    std::unordered_map<std::string, std::function<std::shared_ptr<T>(Args...)>> creators_;
};

template <typename T, typename... Args>
Factory<T, Args...>* Factory<T, Args...>::instance_ = nullptr;