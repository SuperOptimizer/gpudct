cmake_minimum_required(VERSION 3.28)
project(gpudct LANGUAGES CXX VERSION 0.1.0)

# ---------------------------------------------------------------------------
# Standard / options
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)   # fall back to 23 on older toolchains
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPUDCT_BUILD_TESTS  "Build tests"                ON)
option(GPUDCT_BUILD_TOOLS  "Build the gpudct CLI"       ON)
option(GPUDCT_BUILD_BENCH  "Build benchmarks"           ON)
option(GPUDCT_ENABLE_CUDA  "Build the CUDA backend"     OFF)
option(GPUDCT_ENABLE_SIMD  "Build the SIMD CPU kernels" ON)
option(GPUDCT_WERROR       "Treat warnings as errors"   OFF)

# x86 instruction-set baseline.
#
# Defaults to sse2, which is both the portable choice and -- measured -- the fast
# one. An isolated benchmark of the inverse transform showed AVX2 39% ahead, but
# an interleaved A/B of full decode showed no gain at all (110 vs 114 MB/s on a
# real brick): the transform is only ~40% of decode, and whatever AVX2 wins there
# does not survive in context.
#
# It also costs something real. Raising the baseline changed the decoded output
# by 1 LSB on 2 voxels in 2M, so archives stop being reproducible across builds
# of the same source -- which the golden tests exist to guarantee. Paying that
# for no measured speedup is a bad trade.
#
# Per-ISA kernels behind runtime dispatch are the actual answer (docs/ROADMAP.md
# M6). This knob is for experiments until then.
set(GPUDCT_X86_BASELINE "sse2" CACHE STRING "x86 ISA baseline: sse2 | avx2 | native")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Warning / float flags
#
# The scalar reference path must be reproducible byte-for-byte across machines,
# so FP contraction and fast-math are off globally. Optimized kernels opt back
# into contraction locally; docs/QUALITY.md section 4.2 is what holds them honest.
# ---------------------------------------------------------------------------
add_library(gpudct_flags INTERFACE)

# All of these are C++-only. nvcc forwards flags it does not recognize to the
# host compiler, which then chokes on clang spellings like -ffp-contract=off, so
# CUDA sources must not see them at all.
set(_gpudct_cxx_flags "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86|AMD64|amd64)" OR CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(GPUDCT_X86_BASELINE STREQUAL "avx2")
    if(MSVC)
      list(APPEND _gpudct_cxx_flags /arch:AVX2)
    else()
      list(APPEND _gpudct_cxx_flags -mavx2)
    endif()
  elseif(GPUDCT_X86_BASELINE STREQUAL "native")
    if(MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND _gpudct_cxx_flags -march=native)
    elseif(NOT MSVC)
      list(APPEND _gpudct_cxx_flags -march=native)
    endif()
  endif()
endif()

if(MSVC)
  list(APPEND _gpudct_cxx_flags /W4 /permissive- /fp:precise /Zc:preprocessor)
  target_compile_definitions(gpudct_flags INTERFACE _CRT_SECURE_NO_WARNINGS)
  if(GPUDCT_WERROR)
    list(APPEND _gpudct_cxx_flags /WX)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND _gpudct_cxx_flags -ffp-contract=off)
  endif()
else()
  list(APPEND _gpudct_cxx_flags
    -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
    -ffp-contract=off -fno-fast-math)
  if(GPUDCT_WERROR)
    list(APPEND _gpudct_cxx_flags -Werror)
  endif()
endif()
target_compile_options(gpudct_flags INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:${_gpudct_cxx_flags}>")

# ---------------------------------------------------------------------------
# CUDA backend (optional)
#
# Needs a CUDA toolkit. Blackwell parts (RTX 50xx) are sm_120 and want CUDA 13+.
# Enable with -DGPUDCT_ENABLE_CUDA=ON.
# ---------------------------------------------------------------------------
if(GPUDCT_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)          # nvcc tops out at C++20
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80 89 120)
    endif()
    message(STATUS "  CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "GPUDCT_ENABLE_CUDA=ON but no CUDA compiler found; disabling")
    set(GPUDCT_ENABLE_CUDA OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(GPUDCT_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(GPUDCT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(GPUDCT_BUILD_BENCH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
  add_subdirectory(bench)
endif()

message(STATUS "gpudct ${PROJECT_VERSION}")
message(STATUS "  compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++      : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA     : ${GPUDCT_ENABLE_CUDA}")
message(STATUS "  SIMD     : ${GPUDCT_ENABLE_SIMD}")
message(STATUS "  x86 base : ${GPUDCT_X86_BASELINE}")
