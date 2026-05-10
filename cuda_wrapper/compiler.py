"""
CUDA kernel compilation utilities using nvcc.

Wiki: wiki/07-cuda-wrapper.md (section 7.4 "Kernels" and 7.7).
File-locked, dependency-aware nvcc invocation that produces .cubin files
under kernel_dir/compiled/. Recompiles when any quoted-#include header is
newer than the binary.
"""

import subprocess
import filelock
import re
from pathlib import Path


def _get_all_dependencies(cu_file: Path, visited: set = None) -> list[Path]:
    """
    Recursively find all #include'd local files from a .cu or .cuh file.
    Only finds quoted includes (#include "file.cuh"), not system includes (<>).
    
    Returns list of all dependency paths (including the original file).
    """
    if visited is None:
        visited = set()
    
    cu_file = Path(cu_file).resolve()
    if cu_file in visited:
        return []
    visited.add(cu_file)
    
    deps = [cu_file]
    
    if not cu_file.exists():
        return deps
    
    # Regex to match #include "filename"
    include_pattern = re.compile(r'#include\s+"([^"]+)"')
    
    try:
        content = cu_file.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return deps
    
    base_dir = cu_file.parent
    
    for match in include_pattern.finditer(content):
        include_name = match.group(1)
        # Try relative to the current file's directory
        include_path = (base_dir / include_name).resolve()
        
        if include_path.exists() and include_path not in visited:
            deps.extend(_get_all_dependencies(include_path, visited))
    
    return deps


def _get_newest_source_mtime(cu_file: Path) -> float:
    """
    Get the newest modification time among the .cu file and all its #include dependencies.
    """
    newest = 0.0
    for dep in _get_all_dependencies(cu_file):
        try:
            mtime = dep.stat().st_mtime
            if mtime > newest:
                newest = mtime
        except Exception:
            pass
    return newest


def compile_kernel(cu_file: Path, output_file: Path, arch: str, force: bool = False) -> Path:
    """
    Compile a .cu file to cubin using nvcc.
    
    Uses file locking to prevent race conditions when multiple processes
    try to compile the same kernel simultaneously.
    
    Args:
        cu_file: Path to the .cu source file
        output_file: Path for the output .cubin file
        arch: GPU architecture (e.g., "sm_89")
        force: Force recompilation even if cached
        
    Returns:
        Path to the compiled .cubin file
    """
    cu_file = Path(cu_file)
    output_file = Path(output_file)
    lock_file = output_file.with_suffix(".lock")
    
    # Include path for headers (relative includes like "basics/reduction.cuh")
    include_dir = cu_file.parent
    
    # Use file lock to prevent concurrent compilation of the same kernel
    compiled = False
    with filelock.FileLock(lock_file, timeout=60):
        # Check if recompilation is needed (inside lock to avoid races)
        need_compile = force or not output_file.exists()
        
        if not need_compile and output_file.exists():
            # Recompile if ANY source file (.cu or .cuh) is newer than the binary
            newest_source = _get_newest_source_mtime(cu_file)
            if newest_source > output_file.stat().st_mtime:
                need_compile = True
        
        if need_compile:
            compiled = True

            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "nvcc",
                f"-arch={arch}",
                "-cubin",
                f"-I{include_dir}",       # Include path for header files
                # Optimization flags
                "-O3",                    # Maximum host-side optimization level
                "--use_fast_math",        # Fast math: fuses mul+add, relaxes denormals, fast rsqrt/div/sin/cos/exp/log
                "--extra-device-vectorization",  # Aggressive loop vectorization on GPU
                "-o", str(output_file),
                str(cu_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # NOTE: On Windows, nvcc often shells out to cl.exe which may emit
                # errors to stdout (not stderr). Print both so diagnostics are visible.
                stdout = (result.stdout or "").strip()
                stderr = (result.stderr or "").strip()

                print("\n\n\n\n")
                if stdout:
                    print(stdout)
                if stderr:
                    print(stderr)
                if not stdout and not stderr:
                    print("nvcc failed but produced no stdout/stderr output.")
                print("\n\n\n\n")

                details = stderr or stdout or "(no nvcc output)"
                raise RuntimeError(f"nvcc compilation failed for {cu_file}:\n{details}")
    return output_file, compiled
