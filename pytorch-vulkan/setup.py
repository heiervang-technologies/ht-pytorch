"""Build script for pytorch-vulkan (Rust + C++ shim)."""

import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


class RustCppBuildExt(BuildExtension):
    """Custom BuildExtension that first compiles the Rust crate."""

    def build_extension(self, ext):
        rust_dir = Path(__file__).parent / "rust"
        rust_target_dir = rust_dir / "target" / "release"

        # Build Rust crate.
        subprocess.check_call(
            ["cargo", "build", "--release"],
            cwd=str(rust_dir),
        )

        # Add Rust library paths.
        ext.extra_objects.append(str(rust_target_dir / "libvulkan_compute.a"))

        # Generated header location.
        generated_dir = Path(__file__).parent / "csrc" / "generated"
        if generated_dir.exists():
            ext.include_dirs.append(str(generated_dir))

        super().build_extension(ext)


setup(
    name="pytorch-vulkan",
    version="0.1.0",
    packages=["pytorch_vulkan"],
    ext_modules=[
        CppExtension(
            name="pytorch_vulkan._C",
            sources=["csrc/shim.cpp"],
            libraries=["vulkan"],
            extra_compile_args=["-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": RustCppBuildExt},
)
