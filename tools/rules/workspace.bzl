def _impl(repository_ctx):
    archive = repository_ctx.attr.name + ".tar"
    reference = Label("@%s_unpatched//:README" % repository_ctx.attr.name)
    dirname = repository_ctx.path(reference).dirname
    repository_ctx.execute(["tar", "hcf", archive, "-C", dirname, "."])
    repository_ctx.extract(archive)
    for patch in repository_ctx.attr.patches:
        repository_ctx.patch(repository_ctx.path(patch), repository_ctx.attr.patch_strip)
    build_file = repository_ctx.path(repository_ctx.attr.build_file)
    repository_ctx.execute(["cp", build_file, "BUILD.bazel"])

_patched_rule = repository_rule(
    implementation = _impl,
    attrs = {
        "build_file": attr.label(),
        "patch_strip": attr.int(),
        "patches": attr.label_list(),
    },
)

def new_patched_local_repository(name, path, **kwargs):
    native.new_local_repository(
        name = name + "_unpatched",
        build_file_content = """
pkg_tar(name = "content", srcs = glob(["**"]))
""",
        path = path,
    )
    _patched_rule(name = name, **kwargs)

def _new_empty_repository_impl(repo_ctx):
    build_file = repo_ctx.attr.build_file
    build_file_content = repo_ctx.attr.build_file_content
    if not (bool(build_file) != bool(build_file_content)):
        fail("Exactly one of 'build_file' or 'build_file_content' is required")

    if build_file_content:
        repo_ctx.file("BUILD", build_file_content)
    elif build_file:
        repo_ctx.template("BUILD", repo_ctx.attr.build_file, {})

new_empty_repository = repository_rule(
    attrs = {
        "build_file": attr.label(allow_files = True),
        "build_file_content": attr.string(),
    },
    implementation = _new_empty_repository_impl,
)

"""Create an empty repository with the supplied BUILD file.

This is mostly useful to create wrappers for specific target that we want
to be used with the '@' syntax.
"""

def _find_cudnn_impl(repository_ctx):
    cudnn_path = repository_ctx.attr.path
    if not cudnn_path:
        cudnn_path = repository_ctx.os.environ.get("CUDNN_PATH", "")

    if cudnn_path:
        # System cuDNN: symlink the provided path
        repository_ctx.symlink(cudnn_path, "cudnn")
    else:
        # Pip cuDNN: find via python
        python = repository_ctx.which("python3") or repository_ctx.which("python")
        if not python:
            fail("CUDNN_PATH not set and python3 not found for pip fallback")
        result = repository_ctx.execute([
            python, "-c",
            "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])",
        ])
        if result.return_code != 0:
            fail("CUDNN_PATH not set and pip nvidia-cudnn not found: " + result.stderr)
        repository_ctx.symlink(result.stdout.strip(), "cudnn")

    # Detect lib directory (system uses lib64/, pip uses lib/)
    lib_dir = "lib"
    if repository_ctx.path("cudnn/lib64").exists:
        lib_dir = "lib64"
    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "cudnn_headers",
    hdrs = glob(["cudnn/include/cudnn*.h"]),
    includes = ["cudnn/include"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "cudnn/{lib_dir}/libcudnn.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
""".format(lib_dir = lib_dir))

find_cudnn = repository_rule(
    implementation = _find_cudnn_impl,
    attrs = {
        "path": attr.string(doc = "Path to cuDNN. If empty, uses CUDNN_PATH env var, then falls back to pip nvidia-cudnn."),
    },
    environ = ["CUDNN_PATH", "PATH"],
)

def _find_cuda_impl(repository_ctx):
    cuda_path = repository_ctx.attr.path
    if not cuda_path:
        cuda_path = repository_ctx.os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        cuda_path = "/usr/local/cuda"
    if not repository_ctx.path(cuda_path).exists:
        fail("CUDA toolkit not found. Set CUDA_PATH or pass path attribute.")
    repository_ctx.symlink(cuda_path, "cuda")

    # Detect lib/include layout
    if repository_ctx.path("cuda/targets/x86_64-linux/lib").exists:
        lib_dir = "targets/x86_64-linux/lib"
        inc_dirs = ["include", "targets/x86_64-linux/include", "targets/x86_64-linux/include/nvtx3"]
    else:
        lib_dir = "lib64"
        inc_dirs = ["include", "include/nvtx3"]

    inc_globs = "\n".join(['        "cuda/{dir}/**",'.format(dir = d) for d in inc_dirs])
    inc_includes = "\n".join(['        "cuda/{dir}",'.format(dir = d) for d in inc_dirs])

    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "cuda_headers",
    hdrs = glob([
{inc_globs}
    ]),
    includes = [
{inc_includes}
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda/{lib_dir}/stubs/libcuda.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    srcs = ["cuda/{lib_dir}/libcudart.so"],
    visibility = ["//visibility:public"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cufft",
    srcs = ["cuda/{lib_dir}/libcufft.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas",
    srcs = [
        "cuda/{lib_dir}/libcublasLt.so",
        "cuda/{lib_dir}/libcublas.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand",
    srcs = ["cuda/{lib_dir}/libcurand.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusolver",
    srcs = ["cuda/{lib_dir}/libcusolver.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cusparse",
    srcs = ["cuda/{lib_dir}/libcusparse.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufile",
    srcs = ["cuda/{lib_dir}/libcufile.so"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvrtc",
    srcs = glob(["cuda/{lib_dir}/libnvrtc*.so"]),
    visibility = ["//visibility:public"],
)
""".format(lib_dir = lib_dir, inc_globs = inc_globs, inc_includes = inc_includes))

find_cuda = repository_rule(
    implementation = _find_cuda_impl,
    attrs = {
        "path": attr.string(doc = "Path to CUDA toolkit. If empty, uses CUDA_PATH env var, then falls back to /usr/local/cuda."),
    },
    environ = ["CUDA_PATH"],
)
