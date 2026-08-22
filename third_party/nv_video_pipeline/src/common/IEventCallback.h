#pragma once

#include <functional>

namespace GraphCore {

/**
 *  通用的回调函数
 *  (tag, code, msg)-> code
 */
using CallBackFunction = std::function<int()>;

/**
 *  定义了一些回调接口基类
 */
class IEventCallBack {
protected:
    // 错误回调
    CallBackFunction error_cb = nullptr;
    // 处理超时回调
    CallBackFunction timeout_cb = nullptr;
    // 缓冲溢出回调
    CallBackFunction buffer_over_cb = nullptr;
    // 节点启动回调
    CallBackFunction before_start_cb = nullptr;
    CallBackFunction after_start_cb  = nullptr;
    // 节点退出回调
    CallBackFunction exit_cb = nullptr;

public:
    void set_error_cb(CallBackFunction event_cb) {
        error_cb = std::move(event_cb);
    }

    void set_timeout_cb(CallBackFunction event_cb) {
        timeout_cb = std::move(event_cb);
    }

    void set_buffer_over_cb(CallBackFunction event_cb) {
        buffer_over_cb = std::move(event_cb);
    }

    void set_exit_cb(CallBackFunction event_cb) {
        exit_cb = std::move(event_cb);
    }

    void set_after_start_cb(CallBackFunction event_cb) {
        after_start_cb = std::move(event_cb);
    }

    void set_before_start_cb(CallBackFunction event_cb) {
        before_start_cb = std::move(event_cb);
    }
};

}  // namespace GraphCore