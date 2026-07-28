"""Build script for pytorch-vulkan (Rust + C++ shim)."""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension


_shaders_compiled = False


def compile_shaders() -> None:
    global _shaders_compiled
    if _shaders_compiled:
        return
    root = Path(__file__).parent
    subprocess.check_call(
        [sys.executable, str(root / "tools" / "compile_shaders.py")],
        cwd=str(root),
    )
    _shaders_compiled = True


class ShaderBuildPy(build_py):
    def run(self):
        compile_shaders()
        super().run()


class RustCppBuildExt(BuildExtension):
    """Custom BuildExtension that first compiles the Rust crate."""

    def run(self):
        compile_shaders()
        super().run()

    def build_extension(self, ext):
        rust_dir = Path(__file__).parent / "rust"
        rust_target_dir = rust_dir / "target" / "release"

        # Build Rust crate.
        subprocess.check_call(
            ["cargo", "build", "--release", "--locked"],
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
    ext_modules=[
        CppExtension(
            name="pytorch_vulkan._C",
            sources=["csrc/shim.cpp"],
            libraries=["vulkan"],
            extra_compile_args=["-std=c++17"],
        ),
    ],
    cmdclass={
        "build_ext": RustCppBuildExt,
        "build_py": ShaderBuildPy,
    },
)
