// https://github.com/mapillary/inplace_abn/blob/main/include/cuda_utils.cuh
// Copyright (c) Facebook, Inc. and its affiliates.

#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_PROCESSORS 4608
#define MAX_SHARED_MEMORY 16000

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int scr_line, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, value, scr_line, width);
#else
    return __shfl(value, scr_line, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(mask, value, delta, width);
#else
    return __shfl_up(value, delta, width);
#endif
}

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

static int getNumOrder(int n) {
  int NSizes[11] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

  for (int i = 0; i < 11; ++i) {
    if (n <= NSizes[i]) {
      return NSizes[i];
    }
  }
  throw std::runtime_error("num_order must be leq than 1024.");
}

// #define BIGGEST

// Number of threads = BlockDim
// Must be a multiple of WARP_SIZE & leq than MAX_BLOCK_SIZE
static int getBlockDim(int n) {
#ifdef BIGGEST    // set BlockDim as big as possible.
  if (n < WARP_SIZE) {
    #pragma unroll
    for (int bd = MAX_BLOCK_SIZE; bd >= WARP_SIZE; bd -= WARP_SIZE) {
      if (bd % n == 0) return bd;
    }
  }
  else {
    #pragma unroll
    for (int bd = MAX_BLOCK_SIZE; bd >= WARP_SIZE; bd -= WARP_SIZE) {
      if (n % bd == 0) return bd;
    }
  }
#else   // set BlockDim as small as possible.
  if (n < WARP_SIZE) {
    #pragma unroll
    for (int bd = WARP_SIZE; bd <= MAX_BLOCK_SIZE; bd += WARP_SIZE) {
      if (bd % n == 0) return bd;
    }
  }
  else {
    if (n % WARP_SIZE == 0) return WARP_SIZE;
  }
#endif
  return 0;
}