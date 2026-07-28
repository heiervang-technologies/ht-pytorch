from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEFAULT_SOURCE = ROOT / "shaders"
DEFAULT_OUTPUT = ROOT / "pytorch_vulkan" / "_shaders"


def compile_shader(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        command = [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.2",
            str(source),
            "-o",
            str(temporary_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to compile {source.name}:\n{result.stdout}{result.stderr}"
            )
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def compile_all(source_dir: Path, output_dir: Path) -> int:
    sources = sorted(source_dir.glob("*.comp"))
    if not sources:
        raise RuntimeError(f"no compute shaders found in {source_dir}")
    expected = set()
    for source in sources:
        output = output_dir / f"{source.name}.spv"
        expected.add(output.name)
        compile_shader(source, output)
    for stale in output_dir.glob("*.spv"):
        if stale.name not in expected:
            stale.unlink()
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = compile_all(args.source, args.output)
    print(f"compiled {count} Vulkan shaders into {args.output}")


if __name__ == "__main__":
    main()
