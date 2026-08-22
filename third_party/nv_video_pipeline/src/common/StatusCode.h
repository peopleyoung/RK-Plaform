#pragma once

#include "CLog.h"

#define CHECK(condition, message) \
    do {                          \
        if (!(condition)) {       \
            LOG_ERROR(message);   \
            return false;         \
        }                         \
    } while (0)
