#pragma once

#include <Util/TimeTicker.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "Factory.h"
#include "IEventCallback.h"
#include "ThreadSafeDataList.h"

namespace GraphCore {

enum NODE_TYPE {
    SRC_NODE,  // 输入节点
    MID_NODE,  // 中间节点
    DES_NODE,  // 输出节点
    UNK_NODE,  // 未知节点
};

class Node : public IEventCallBack {
public:
    using ptr                 = std::shared_ptr<Node>;
    using QUEUE               = ThreadSafeDataList::ptr;
    using ExtraInputCallBack  = std::function<void(Data::BaseData::ptr)>;
    using DataHandleCallBack  = std::function<Data::BaseData::ptr(Data::BaseData::ptr)>;
    using BatchDataHookerFunc = std::function<std::vector<Data::BaseData::ptr>(std::vector<Data::BaseData::ptr> &)>;

    Node() = delete;

    explicit Node(const std::string &name) : m_name(std::move(name)) {
    }

    ~Node();

public:
    /**
     * @brief 初始化节点
     */

    virtual bool Init(YAML::Node config);

    /**
     * @brief 启动工作线程
     */
    void start();

    /**
     * @brief 停止线程
     */
    void stop();

    const std::string getName();

    NODE_TYPE getType();

    void add_input(const std::string &tag, QUEUE queue);

    /**
     * @brief
     * 给当前节点添加输入缓冲队列，所有的输入缓冲都共享该同一个条件变量，因此N个输入节点的任意数据输入都会唤醒当前堵塞的worker线程
     * @param tag
     * @param queue
     */
    void add_output(const std::string &tag, QUEUE queue);

    void del_input(const std::string &tag);

    void del_output(const std::string &tag);

    /**
     * @brief 向节点的输入队列中添加数据，主要是用来输入控制和配置信息
     * @param data
     */
    void add_data(const Data::BaseData::ptr &data);

    void add_datas(const std::vector<Data::BaseData::ptr> &datas);

    void set_get_data_max_num(int num);

    std::tuple<double, float> get_perf();

public:
    void set_extra_input_callback(ExtraInputCallBack callback);

    void add_extra_data(const Data::BaseData::ptr &data);

    void set_before_data_handle_callback(DataHandleCallBack callback);
    void set_after_data_handle_callback(DataHandleCallBack callback);
    void set_batch_data_handler_hooker(BatchDataHookerFunc batch_data_hooker);

protected:
    void worker();

    /**
     * @brief 从输入队列中获取数据，如果队列为空则阻塞等待
     * @param datas
     */
    void get_input_datas(std::vector<Data::BaseData::ptr> &datas);

    void send_output_data(const Data::BaseData::ptr &data);

    void send_output_datas(const std::vector<Data::BaseData::ptr> &datas);

    /**
     * @brief 处理数据业务接口
     * @param data 所有数据的继承基类
     * @return
     */
    virtual Data::BaseData::ptr handle_data(Data::BaseData::ptr data);

protected:
    ExtraInputCallBack  m_extra_input_callback        = nullptr;
    DataHandleCallBack  m_before_data_handle_callback = nullptr;
    DataHandleCallBack  m_after_data_handle_callback  = nullptr;
    BatchDataHookerFunc batch_data_handler_hooker     = nullptr;

protected:
    const std::string                        m_name;
    std::thread                              m_worker;
    NODE_TYPE                                m_type             = UNK_NODE;
    bool                                     m_run              = false;
    int                                      m_get_data_max_num = 1;
    std::mutex                               m_base_mutex;
    std::mutex                               m_perf_mutex;
    double                                   m_perf_time  = 0;
    float                                    m_perf_count = 0;
    toolkit::Ticker                          m_ticker;
    std::atomic_int                          m_buffer_over_count = 0;
    std::shared_ptr<std::condition_variable> m_base_cond         = std::make_shared<std::condition_variable>();

    std::map<std::string, QUEUE> m_input_buffers;
    std::map<std::string, QUEUE> m_output_buffers;
};

/**
 * @brief 将两个节点连接起来
 * @param front  前一个节点
 * @param back   后一个节点
 * @param max_cache  缓冲队列最大缓存帧数,默认25
 * @param strategy   缓冲队列满时的策略，默认丢弃最早的帧
 */
static inline void LinkNode(const Node::ptr &front, const Node::ptr &back, int max_cache = 50,
                            BufferOverStrategy strategy = BufferOverStrategy::DROP_EARLY) {
    auto queue = std::make_shared<ThreadSafeDataList>(back->getName());
    queue->set_max_size(max_cache);
    queue->set_buffer_strategy(strategy);
    back->add_input(front->getName(), queue);
    front->add_output(back->getName(), queue);
}

static inline void UnLinkNode(const Node::ptr &front, const Node::ptr &back) {
    front->del_output(back->getName());
    back->del_input(front->getName());
}

using NodesFactory = Factory<Node, std::string>;

}  // namespace GraphCore
