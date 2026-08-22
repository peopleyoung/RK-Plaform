#include "MediaUrl.h"

#include <cassert>
#include <string>

int main() {
    const std::string sentinel = "must-not-appear";
    const std::string redacted = media::redact_media_url(
        "rtsp://192.0.2.10:8554/live/camera_01?publishToken=" + sentinel
        + "#fragment");
    assert(redacted == "rtsp://192.0.2.10:8554/live/camera_01");
    assert(redacted.find(sentinel) == std::string::npos);
    assert(media::redact_media_url("rtsp://[2001:db8::1]:8554/live/camera")
           == "rtsp://[2001:db8::1]:8554/live/camera");
    return 0;
}
