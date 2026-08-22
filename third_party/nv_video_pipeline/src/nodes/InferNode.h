#pragma once

#include <atomic>
#include <memory>

#include "BaseData.h"
#include "FrameTarget.h"
#include "Instance.h"
#include "ProcessNode.h"
#include "Thread/ThreadPool.h"
#include "ThreadsafeQueue.h"

namespace Node {

class InferNode : public GraphCore::Node {
public:
    using ptr = std::shared_ptr<InferNode>;

    explicit InferNode(const std::string& name);

    void set_instance(infer::Instance::ptr instance);
    bool Init(YAML::Node config) override;

private:
    Data::BaseData::ptr handle_data(Data::BaseData::ptr data) override;
    Data::BaseData::ptr handle_data_interval(Data::BaseData::ptr data);
    Data::BaseData::ptr handle_data_normal(Data::BaseData::ptr data);

    utils::ThreadsafeQueue<object_meta::FrameTargetList::ptr>::ptr m_target_queue;
    std::atomic_int                                                m_target_in_queue = 0;
    int                                                            after_start_cb();
    int                                                            exit_cb();

private:
    infer::Instance::ptr                 m_instance;
    uint32_t                             m_interval = 1;
    std::string                          m_instance_name;
    std::shared_ptr<toolkit::ThreadPool> pool;
    std::atomic_int                      m_commit_error_count = 0;
};

}  // namespace Node
