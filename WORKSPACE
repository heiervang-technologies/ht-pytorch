workspace(name = "pytorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools/rules:workspace.bzl", "find_cuda", "find_cudnn", "new_patched_local_repository")

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

http_archive(
    name = "com_github_glog",
    build_file_content = """
licenses(['notice'])

load(':bazel/glog.bzl', 'glog_library')
# TODO: figure out why enabling gflags leads to SIGSEV on the logging init
glog_library(with_gflags=0)
    """,
    strip_prefix = "glog-0.4.0",
    urls = [
        "https://github.com/google/glog/archive/v0.4.0.tar.gz",
    ],
)

http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)

new_local_repository(
    name = "gloo",
    build_file = "//third_party:gloo.BUILD",
    path = "third_party/gloo",
)

new_local_repository(
    name = "onnx",
    build_file = "//third_party:onnx.BUILD",
    path = "third_party/onnx",
)

new_local_repository(
    name = "cutlass",
    build_file = "//third_party:cutlass.BUILD",
    path = "third_party/cutlass",
)

new_local_repository(
    name = "fbgemm",
    build_file = "//third_party:fbgemm/BUILD.bazel",
    path = "third_party/fbgemm",
    repo_mapping = {"@cpuinfo": "@org_pytorch_cpuinfo"},
)

new_local_repository(
    name = "ideep",
    build_file = "//third_party:ideep.BUILD",
    path = "third_party/ideep",
)

new_local_repository(
    name = "mkl_dnn",
    build_file = "//third_party:mkl-dnn.BUILD",
    path = "third_party/ideep/mkl-dnn",
)

new_local_repository(
    name = "org_pytorch_cpuinfo",
    build_file = "//third_party:cpuinfo/BUILD.bazel",
    path = "third_party/cpuinfo",
)

new_local_repository(
    name = "asmjit",
    build_file = "//third_party:fbgemm/external/asmjit.BUILD",
    path = "third_party/fbgemm/external/asmjit",
)

new_local_repository(
    name = "sleef",
    build_file = "//third_party:sleef.BUILD",
    path = "third_party/sleef",
)

new_local_repository(
    name = "fmt",
    build_file = "//third_party:fmt.BUILD",
    path = "third_party/fmt",
)

new_local_repository(
    name = "kineto",
    build_file = "//third_party:kineto.BUILD",
    path = "third_party/kineto",
)

new_local_repository(
    name = "cpp-httplib",
    build_file = "//third_party:cpp-httplib.BUILD",
    path = "third_party/cpp-httplib",
)

new_local_repository(
    name = "nlohmann",
    build_file = "//third_party:nlohmann.BUILD",
    path = "third_party/nlohmann",
)

new_local_repository(
    name = "moodycamel",
    build_file = "//third_party:moodycamel.BUILD",
    path = "third_party/concurrentqueue",
)

new_local_repository(
    name = "tensorpipe",
    build_file = "//third_party:tensorpipe.BUILD",
    path = "third_party/tensorpipe",
)

http_archive(
    name = "mkl",
    build_file = "//third_party:mkl.BUILD",
    sha256 = "59154b30dd74561e90d547f9a3af26c75b6f4546210888f09c9d4db8f4bf9d4c",
    strip_prefix = "lib",
    urls = [
        "https://anaconda.org/anaconda/mkl/2020.0/download/linux-64/mkl-2020.0-166.tar.bz2",
    ],
)

http_archive(
    name = "mkl_headers",
    build_file = "//third_party:mkl_headers.BUILD",
    sha256 = "2af3494a4bebe5ddccfdc43bacc80fcd78d14c1954b81d2c8e3d73b55527af90",
    urls = [
        "https://anaconda.org/anaconda/mkl-include/2020.0/download/linux-64/mkl-include-2020.0-166.tar.bz2",
    ],
)


find_cuda(name = "cuda")

find_cudnn(name = "cudnn")

new_local_repository(
    name = "cudnn_frontend",
    build_file = "@//third_party:cudnn_frontend.BUILD",
    path = "third_party/cudnn_frontend/",
)

local_repository(
    name = "com_github_google_flatbuffers",
    path = "third_party/flatbuffers",
)

local_repository(
    name = "google_benchmark",
    path = "third_party/benchmark",
)

local_repository(
    name = "com_google_googletest",
    path = "third_party/googletest",
)

local_repository(
    name = "pthreadpool",
    path = "third_party/pthreadpool",
    repo_mapping = {"@com_google_benchmark": "@google_benchmark"},
)

local_repository(
    name = "FXdiv",
    path = "third_party/FXdiv",
    repo_mapping = {"@com_google_benchmark": "@google_benchmark"},
)

local_repository(
    name = "XNNPACK",
    path = "third_party/XNNPACK",
    repo_mapping = {"@com_google_benchmark": "@google_benchmark"},
)

local_repository(
    name = "gemmlowp",
    path = "third_party/gemmlowp/gemmlowp",
)

local_repository(
    name = "kleidiai",
    path = "third_party/kleidiai",
    repo_mapping = {"@com_google_googletest": "@com_google_benchmark"},
)

### Unused repos start

# `unused` repos are defined to hide bazel files from submodules of submodules.
# This allows us to run `bazel build //...` and not worry about the submodules madness.
# Otherwise everything traverses recursively and a lot of submodules of submodules have
# they own bazel build files.

local_repository(
    name = "unused_tensorpipe_googletest",
    path = "third_party/tensorpipe/third_party/googletest",
)

local_repository(
    name = "unused_fbgemm",
    path = "third_party/fbgemm",
)

local_repository(
    name = "unused_ftm_bazel",
    path = "third_party/fmt/support/bazel",
)

local_repository(
    name = "unused_kineto_fmt_bazel",
    path = "third_party/kineto/libkineto/third_party/fmt/support/bazel",
)

local_repository(
    name = "unused_kineto_dynolog_googletest",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/googletest",
)

local_repository(
    name = "unused_kineto_dynolog_gflags",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/gflags",
)

local_repository(
    name = "unused_kineto_dynolog_glog",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/glog",
)

local_repository(
    name = "unused_kineto_googletest",
    path = "third_party/kineto/libkineto/third_party/googletest",
)

local_repository(
    name = "unused_onnx_benchmark",
    path = "third_party/onnx/third_party/benchmark",
)

### Unused repos end
