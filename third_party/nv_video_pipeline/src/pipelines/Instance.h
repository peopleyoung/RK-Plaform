#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <tuple>

#include "BaseData.h"
#include "Factory.h"

namespace infer {

class Instance {
public:
    using ptr = std::shared_ptr<Instance>;
    explicit Instance(const std::string &name) : m_infer_name(name){};
    virtual ~Instance() = default;

public:
    virtual bool init(YAML::Node config) = 0;
    virtual bool commit(Job &job)        = 0;
    virtual void start()                 = 0;
    virtual void stop()                  = 0;

    // Interval reuse is only valid for result types with a cheap, immutable
    // frame snapshot. Structured segmentation/OCR results run every frame.
    virtual bool supports_interval_reuse() const {
        return false;
    }

    const std::string getName() const {
        return m_infer_name;
    }

    virtual std::tuple<double, float> get_perf() = 0;

protected:
    const std::string m_infer_name;
};

using InstanceFactory = Factory<Instance, std::string>;

}  // namespace infer
