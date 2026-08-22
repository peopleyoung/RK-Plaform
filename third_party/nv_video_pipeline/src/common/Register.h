#pragma once

#include <memory>

#include "Factory.h"

template <typename Base, typename Impl, typename... Args>
class Register {
public:
    explicit Register(const std::string& name) {
        Factory<Base, Args...>& factory = Factory<Base, Args...>::Instance();
        factory.Register(name, [](Args... args) {
            return std::shared_ptr<Base>(new Impl(args...));
        });
    }
};