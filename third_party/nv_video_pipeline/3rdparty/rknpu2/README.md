# RKNN Runtime SDK

This directory vendors the minimum RKNN 2.3.2 runtime subset required by the
RK3588 C++ backend. The files were copied from the board's matching
`rknn_model_zoo/3rdparty/rknpu2` snapshot:

```text
3rdparty/rknpu2/include/rknn_api.h
3rdparty/rknpu2/Linux/aarch64/librknnrt.so
3rdparty/rknpu2/LICENSE
```

Validated checksums:

```text
c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a  include/rknn_api.h
d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8  Linux/aarch64/librknnrt.so
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  LICENSE
```

The header, runtime, Toolkit2 converter, board runtime/server, and NPU driver
form one compatibility stack. When upgrading, replace and validate them
together; do not mix files from different RKNN releases. The validated target
baseline is RKNN Toolkit2/runtime/server 2.3.2 with driver 0.9.8 on RK3588.
