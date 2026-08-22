#include "MediaUrl.h"

#include <algorithm>

namespace media {

std::string redact_media_url(const std::string& url) {
    const auto query = url.find('?');
    const auto fragment = url.find('#');
    const auto end = std::min(query == std::string::npos ? url.size() : query,
                              fragment == std::string::npos ? url.size() : fragment);
    return url.substr(0, end);
}

}  // namespace media
