/*
 * This file is part of hipCPU, a HIP implementation based on OpenMP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef CPUCUDA_RUNTIME_H
#define CPUCUDA_RUNTIME_H

#define __CPUCUDA__


#include <cstddef>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "detail/runtime.hpp"



cudaError_t cudaMalloc(void** ptr, size_t size)
{
  *ptr = cudacpu::detail::aligned_malloc(cudacpu::detail::default_alignment, size);

  if(*ptr == nullptr)
    return cudaErrorMemoryAllocation;

  return cudaSuccess;
}

//cudaError_t cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
//cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);


cudaError_t cudaFree(void* ptr)
{
  cudacpu::detail::aligned_free(ptr);
  return cudaSuccess;
}


cudaError_t cudaMallocHost(void** ptr, size_t size)
{
  return cudaMalloc(ptr, size);
}

#define cudaMemAttachGlobal 0
#define cudaMemAttachHost 1

template<class T>

cudaError_t cudaMallocManaged(T** ptr, size_t size, unsigned flags = cudaMemAttachGlobal)
{
  return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}


cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
{
  return cudaMalloc(ptr, size);
}


cudaError_t cudaHostMalloc(void** ptr, size_t size, unsigned int flags)
{
  return cudaMalloc(ptr, size);
}

/*
cudaError_t cudaMallocArray(cudaArray** array,
                                        const cudaChannelFormatDesc* desc,
                                        size_t width, size_t height,
                                        unsigned int flags);
cudaError_t cudaMalloc3DArray(cudaArray** array, const struct cudaChannelFormatDesc* desc,
                            struct cudaExtent extent, unsigned int flags);
cudaError_t cudaFreeArray(cudaArray* array);
cudaError_t cudaHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);*/


cudaError_t cudaFreeHost(void* ptr)
{
  return cudaFree(ptr);
}


cudaError_t cudaHostFree(void* ptr)
{
  return cudaFree(ptr);
}


cudaError_t cudaSetDevice(int device)
{
  if(device != 0)
    return cudaErrorInvalidDevice;

  _cudacpu_runtime.set_device(device);
  return cudaSuccess;
}

//cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp_t* prop);

cudaError_t cudaStreamCreate(cudaStream_t* stream)
{
  *stream = _cudacpu_runtime.create_blocking_stream();
  return cudaSuccess;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
{
  if(flags == cudaStreamDefault)
    return cudaStreamCreate(stream);
  else if (flags == cudaStreamNonBlocking) 
  {
    *stream = _cudacpu_runtime.create_async_stream();
    return cudaSuccess;
  }

  return cudaErrorInvalidValue;
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
  _cudacpu_runtime.streams().get(stream)->wait();
  return cudaSuccess;
}


cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
  _cudacpu_runtime.destroy_stream(stream);
  return cudaSuccess;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                            unsigned int flags)
{
  std::shared_ptr<cudacpu::event> evt = _cudacpu_runtime.events().get_shared(event);
  _cudacpu_runtime.submit_operation([evt](){
    // TODO store error code
    evt->wait();
  }, stream);
  return cudaSuccess;
}


cudaError_t cudaStreamQuery(cudaStream_t stream)
{
  cudacpu::stream* s = _cudacpu_runtime.streams().get(stream);
  
  if(s->is_idle())
    return cudaSuccess;

  return cudaErrorNotReady;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                cudaStreamCallback_t callback, void *userData,
                                unsigned int flags) 
{
  _cudacpu_runtime.submit_operation([stream, callback, userData](){
    // TODO guarantee correct error propagation
    callback(stream, cudaSuccess, userData);
  }, stream);
  return cudaSuccess;
}


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          cudaMemcpyKind copyKind, cudaStream_t stream = 0)
{
  if(!_cudacpu_runtime.streams().is_valid(stream))
    return cudaErrorInvalidValue;

  _cudacpu_runtime.submit_operation([=](){
    memcpy(dst, src, sizeBytes);
  }, stream);
  
  return cudaSuccess;
}

                                            
cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                                   cudaMemcpyKind copyKind)
{
  cudaMemcpyAsync(dst, src, sizeBytes, copyKind, 0);
  _cudacpu_runtime.streams().get(0)->wait();
  return cudaSuccess;
}


cudaError_t cudaMemcpyHtoD(cudaDeviceptr_t dst, void* src, size_t size)
{
  return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}


cudaError_t cudaMemcpyDtoH(void* dst, cudaDeviceptr_t src, size_t size)
{
  return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}


cudaError_t cudaMemcpyDtoD(cudaDeviceptr_t dst, cudaDeviceptr_t src, size_t size)
{
  return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}


cudaError_t cudaMemcpyHtoDAsync(cudaDeviceptr_t dst, void* src, size_t size,
                                            cudaStream_t stream)
{
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}


cudaError_t cudaMemcpyDtoHAsync(void* dst, cudaDeviceptr_t src, size_t size,
                                            cudaStream_t stream)
{
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}


cudaError_t cudaMemcpyDtoDAsync(cudaDeviceptr_t dst, cudaDeviceptr_t src, size_t size,
                                            cudaStream_t stream)
{
  return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
}


cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  cudaMemcpyKind copyType,
                                  cudaStream_t stream = 0)
{
  char* base_ptr = static_cast<char*>(const_cast<void*>(symbol));
  void* ptr = static_cast<void*>(base_ptr + offset);
  return cudaMemcpyAsync(ptr, src, sizeBytes, copyType, stream);
}


cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                    size_t sizeBytes, size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream = 0)
{
  const void* ptr = 
    static_cast<const void*>(static_cast<const char*>(symbolName)+offset);
  return cudaMemcpyAsync(dst, ptr, sizeBytes, kind, stream);
}


cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                            size_t offset = 0,
                            cudaMemcpyKind copyType = cudaMemcpyHostToDevice)
{
  cudaError_t err = 
    cudaMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, copyType, 0);

  if(err != cudaSuccess)
    return err;

  _cudacpu_runtime.streams().get(0)->wait();
  return err;
}


cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbolName,
                               size_t sizeBytes, size_t offset = 0,
                               cudaMemcpyKind kind = cudaMemcpyDeviceToHost) 
{
  cudaError_t err = 
    cudaMemcpyFromSymbolAsync(dst, symbolName, sizeBytes, offset, kind, 0);
    
  if(err != cudaSuccess)
    return err;

  _cudacpu_runtime.streams().get(0)->wait();
  return err;
}


cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);


cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, cudaMemcpyKind kind,
                                          cudaStream_t stream)
{
  if(!_cudacpu_runtime.streams().is_valid(stream))
    return cudaErrorInvalidValue;

  _cudacpu_runtime.submit_operation([=](){
    for(size_t row = 0; row < height; ++row)
    {
      void* row_dst_begin = reinterpret_cast<char*>(dst) + row * dpitch;
      const void* row_src_begin = reinterpret_cast<const char*>(src) + row * spitch;

      memcpy(row_dst_begin, row_src_begin, width);
    }
  }, stream);
  
  return cudaSuccess;
}


cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, cudaMemcpyKind kind)
{
  cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 0);

  if(err != cudaSuccess)
    return err;

  _cudacpu_runtime.streams().get(0)->wait();
  return err;
}

cudaError_t cudaMemcpy2DToArray(cudaArray* dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t spitch, size_t width,
                                            size_t height, cudaMemcpyKind kind);

cudaError_t cudaMemcpyToArray(cudaArray* dst, size_t wOffset, size_t hOffset,
                                          const void* src, size_t count, cudaMemcpyKind kind);

cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t srcArray, size_t wOffset,
                                            size_t hOffset, size_t count, cudaMemcpyKind kind);

cudaError_t cudaMemcpyAtoH(void* dst, cudaArray* srcArray, size_t srcOffset,
                                       size_t count);

cudaError_t cudaMemcpyHtoA(cudaArray* dstArray, size_t dstOffset, const void* srcHost,
                                       size_t count);


cudaError_t cudaDeviceSynchronize()
{
  _cudacpu_runtime.streams().for_each([](cudacpu::stream* s){
    s->wait();
  });
  return cudaSuccess;
}

cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache_t* pCacheConfig);

const char* cudaGetErrorString(cudaError_t error);

const char* cudaGetErrorName(cudaError_t error);


cudaError_t cudaGetDeviceCount(int* count)
{
  *count = 1;
  return cudaSuccess;
}


cudaError_t cudaGetDevice(int* device)
{
  *device = 0;
  return cudaSuccess;
}

/*
cudaError_t cudaIpcCloseMemHandle(void* devPtr);

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event);

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr);

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle);

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle,
                                             unsigned int flags);
*/

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                                        cudaStream_t stream = 0)
{
  if(!_cudacpu_runtime.streams().is_valid(stream))
    return cudaErrorInvalidValue;
  
  _cudacpu_runtime.submit_operation([=](){
    memset(devPtr, value, count);
  }, stream);

  return cudaSuccess;
}


cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
  cudaError_t err = cudaMemsetAsync(devPtr, value, count, 0);
  if(err != cudaSuccess)
    return err;

  _cudacpu_runtime.streams().get(0)->wait();
  return cudaSuccess;
}


cudaError_t cudaMemsetD8(cudaDeviceptr_t dest, unsigned char value, size_t sizeBytes)
{
  return cudaMemset(dest, value, sizeBytes);
}

/*
cudaError_t cudaMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);

cudaError_t cudaMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0);

cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent );

cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent, cudaStream_t stream = 0);
*/


cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t* p_prop, int device)
{
  if(device != 0)
    return cudaErrorInvalidDevice;

  static const char device_name[] = "cudaCPU OpenMP host device";
  int max_dim = std::numeric_limits<int>::max();

  static_assert(sizeof device_name <= sizeof p_prop->name);
  memcpy(p_prop->name, device_name, sizeof device_name);

  // TODO: Find available memory
  p_prop->totalGlobalMem = std::numeric_limits<size_t>::max();
  p_prop->sharedMemPerBlock = _cudacpu_runtime.dev().get_max_shared_memory();
  p_prop->regsPerBlock = std::numeric_limits<int>::max();
  p_prop->warpSize = 1;
  p_prop->maxThreadsPerBlock = _cudacpu_runtime.dev().get_max_threads();
  p_prop->maxGridSize[0] = max_dim;
  p_prop->maxGridSize[1] = max_dim;
  p_prop->maxGridSize[2] = max_dim;
  p_prop->maxGridSize[0] = max_dim;
  p_prop->maxGridSize[1] = max_dim;
  p_prop->maxGridSize[2] = max_dim;
  // TODO: Find actual value
  p_prop->clockRate = 1;
  p_prop->memoryClockRate = 1;
  p_prop->memoryBusWidth = 1;
  p_prop->totalConstMem = std::numeric_limits<std::size_t>::max();
  p_prop->major = 1;
  p_prop->minor = 0;
  p_prop->multiProcessorCount = _cudacpu_runtime.dev().get_num_compute_units();
  // TODO: Find actual value
  p_prop->l2CacheSize = std::numeric_limits<int>::max();
  p_prop->maxThreadsPerMultiProcessor = p_prop->maxThreadsPerBlock;
  p_prop->computeMode = 0;
  p_prop->clockInstructionRate = p_prop->clockRate;

  cudaDeviceArch_t arch;
  arch.hasGlobalInt32Atomics = 1;
  arch.hasGlobalFloatAtomicExch = 1;
  arch.hasSharedInt32Atomics = 1;
  arch.hasSharedFloatAtomicExch = 1;
  arch.hasFloatAtomicAdd = 1;
  arch.hasGlobalInt64Atomics = 1;
  arch.hasSharedInt64Atomics = 1;
  arch.hasDoubles = 1;
  arch.hasWarpVote = 0;
  arch.hasWarpBallot = 0;
  arch.hasWarpShuffle = 0;
  arch.hasFunnelShift = 0;
  arch.hasThreadFenceSystem = 1;
  arch.hasSyncThreadsExt = 1;
  arch.hasSurfaceFuncs = 0;
  arch.has3dGrid = 1;
  arch.hasDynamicParallelism = 0;

  p_prop->arch = arch;
  p_prop->concurrentKernels = 1;
  p_prop->pciBusID = 0;
  p_prop->pciDeviceID = 0;
  p_prop->maxSharedMemoryPerMultiProcessor = p_prop->sharedMemPerBlock;
  p_prop->isMultiGpuBoard = 0;
  p_prop->canMapHostMemory = 1;
  p_prop->gcnArch = 0;
  
  return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttribute_t attr, int device);

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                        const void* func,
                                                        int blockSize,
                                                        size_t dynamicSMemSize);

cudaError_t cudaPointerGetAttributes(cudaPointerAttribute_t* attributes, void* ptr);

cudaError_t cudaMemGetInfo(size_t* free, size_t* total);


cudaError_t cudaEventCreate(cudaEvent_t* event)
{
  *event = _cudacpu_runtime.create_event();
  return cudaSuccess;
}


cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)
{
  if(!_cudacpu_runtime.events().is_valid(event) ||
     !_cudacpu_runtime.streams().is_valid(stream))
    return cudaErrorInvalidValue;

  std::shared_ptr<cudacpu::event> evt = _cudacpu_runtime.events().get_shared(event);
  _cudacpu_runtime.submit_operation([evt](){
    evt->mark_as_finished();
  }, stream);
  return cudaSuccess;
}


cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
  if(!_cudacpu_runtime.events().is_valid(event))
    return cudaErrorInvalidValue;

  cudacpu::event* evt = _cudacpu_runtime.events().get(event);
  evt->wait();

  if(evt->is_complete())
    return cudaSuccess;

  return cudaErrorUnknown;
}


cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop)
{
  if(!_cudacpu_runtime.events().is_valid(start) || !_cudacpu_runtime.events().is_valid(stop))
    return cudaErrorInvalidValue;

  cudacpu::event* start_evt = _cudacpu_runtime.events().get(start);
  cudacpu::event* stop_evt = _cudacpu_runtime.events().get(stop);
  if(start_evt->is_complete() && stop_evt->is_complete())
  {
    *ms = static_cast<float>(stop_evt->timestamp_ns() - start_evt->timestamp_ns()) / 1e6f;
    return cudaSuccess;
  }

  return cudaErrorUnknown;
}


cudaError_t cudaEventDestroy(cudaEvent_t event)
{
  if(!_cudacpu_runtime.events().is_valid(event))
    return cudaErrorInvalidValue;

  _cudacpu_runtime.destroy_event(event);
  return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* driverVersion);


cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
  *runtimeVersion = 99999;
  return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaCtxDisablePeerAccess(cudaCtx_t peerCtx);

cudaError_t cudaCtxEnablePeerAccess(cudaCtx_t peerCtx, unsigned int flags);

cudaError_t cudaDevicePrimaryCtxGetState(cudaDevice_t dev, unsigned int* flags,
                                                     int* active);

cudaError_t cudaDevicePrimaryCtxRelease(cudaDevice_t dev);

cudaError_t cudaDevicePrimaryCtxRetain(cudaCtx_t* pctx, cudaDevice_t dev);

cudaError_t cudaDevicePrimaryCtxReset(cudaDevice_t dev);

cudaError_t cudaDevicePrimaryCtxSetFlags(cudaDevice_t dev, unsigned int flags);

cudaError_t cudaMemGetAddressRange(cudaDeviceptr_t* pbase, size_t* psize,
                                               cudaDeviceptr_t dptr);

cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                                       size_t count);

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count,
                                            cudaStream_t stream = 0);

// Profile APIs:
cudaError_t cudaProfilerStart();
cudaError_t cudaProfilerStop();

cudaError_t cudaSetDeviceFlags(unsigned int flags);

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);


cudaError_t cudaEventQuery(cudaEvent_t event)
{
  if(!_cudacpu_runtime.events().is_valid(event))
    return cudaErrorInvalidValue;

  bool is_ready = _cudacpu_runtime.events().get(event)->is_complete();

  if(!is_ready)
    return cudaErrorNotReady;
  return cudaSuccess;
}

/*
cudaError_t cudaCtxCreate(cudaCtx_t* ctx, unsigned int flags, cudaDevice_t device);

cudaError_t cudaCtxDestroy(cudaCtx_t ctx);

cudaError_t cudaCtxPopCurrent(cudaCtx_t* ctx);

cudaError_t cudaCtxPushCurrent(cudaCtx_t ctx);

cudaError_t cudaCtxSetCurrent(cudaCtx_t ctx);

cudaError_t cudaCtxGetCurrent(cudaCtx_t* ctx);

cudaError_t cudaCtxGetDevice(cudaDevice_t* device);

cudaError_t cudaCtxGetApiVersion(cudaCtx_t ctx, int* apiVersion);

cudaError_t cudaCtxGetCacheConfig(cudaFuncCache* cacheConfig);

cudaError_t cudaCtxSetCacheConfig(cudaFuncCache cacheConfig);

cudaError_t cudaCtxSetSharedMemConfig(cudaSharedMemConfig config);

cudaError_t cudaCtxGetSharedMemConfig(cudaSharedMemConfig* pConfig);

cudaError_t cudaCtxSynchronize(void);

cudaError_t cudaCtxGetFlags(unsigned int* flags);

cudaError_t cudaCtxDetach(cudaCtx_t ctx);

cudaError_t cudaDeviceGet(cudaDevice_t* device, int ordinal);

cudaError_t cudaDeviceComputeCapability(int* major, int* minor, cudaDevice_t device);

cudaError_t cudaDeviceGetName(char* name, int len, cudaDevice_t device);

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, cudaDevice_t device);

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId);

cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* config);

cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);

cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit_t limit);

cudaError_t cudaDeviceTotalMem(size_t* bytes, cudaDevice_t device);

cudaError_t cudaModuleLoad(cudaModule_t* module, const char* fname);

cudaError_t cudaModuleUnload(cudaModule_t hmod);

cudaError_t cudaModuleGetFunction(cudaFunction_t* function, cudaModule_t module,
                                              const char* kname);

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);

cudaError_t cudaModuleGetGlobal(cudaDeviceptr_t* dptr, size_t* bytes, cudaModule_t hmod,
                                            const char* name);

cudaError_t cudaModuleLoadData(cudaModule_t* module, const void* image);

cudaError_t cudaModuleLoadDataEx(cudaModule_t* module, const void* image,
                                             unsigned int numOptions, cudaJitOption* options,
                                             void** optionValues);

cudaError_t cudaModuleLaunchKernel(cudaFunction_t f, unsigned int gridDimX,
                                               unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               cudaStream_t stream, void** kernelParams,
                                               void** extra);


cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache_t cacheConfig);
*/

template <class T>
cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0,
                                                           unsigned int flags = 0);

/*
template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX);

template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, const struct cudaChannelFormatDesc& desc,
                                        size_t size = UINT_MAX);

template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaUnbindTexture(struct texture<T, dim, readMode>* tex);

cudaError_t cudaBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                                        const cudaChannelFormatDesc* desc, size_t size = UINT_MAX);

template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                               cudaArray_const_t array,
                                               const struct cudaChannelFormatDesc& desc);

template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaBindTextureToArray(struct texture<T, dim, readMode> *tex,
                                               cudaArray_const_t array,
                                               const struct cudaChannelFormatDesc* desc);

template <class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t cudaBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                               cudaArray_const_t array);

template <class T>
cudaChannelFormatDesc cudaCreateChannelDesc();

cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                                        cudaChannelFormatKind f);

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                                                const cudaResourceDesc* pResDesc,
                                                const cudaTextureDesc* pTexDesc,
                                                const cudaResourceViewDesc* pResViewDesc);

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t textureObject);

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                                                const cudaResourceDesc* pResDesc);

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfaceObject);

cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc,
                                           cudaTextureObject_t textureObject);

cudaError_t cudaGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array);
*/


__device__

float __fadd_rd(float x, float y)
{
  return x+y;
}

__device__

float __fadd_rn(float x, float y)
{
  return x+y;
}

__device__

float __fadd_ru(float x, float y)
{
  return x+y;
}

__device__

float __fadd_rz(float x, float y)
{
  return x+y;
}

__device__

float __fdiv_rd(float x, float y)
{
  return x/y;
}

__device__

float __fdiv_rn(float x, float y)
{
  return x/y;
}

__device__

float __fdiv_ru(float x, float y)
{
  return x/y;
}

__device__

float __fdiv_rz(float x, float y)
{
  return x/y;
}

__device__

float __fdividef(float x, float y)
{
  return x/y;
}

__device__

float __fmaf_rd(float x, float y, float z)
{
  return std::fma(x,y,z);
}

__device__

float __fmaf_rn(float x, float y, float z)
{
  return std::fma(x,y,z);
}

__device__

float __fmaf_ru(float x, float y, float z)
{
  return std::fma(x,y,z);
}

__device__

float __fmaf_rz(float x, float y, float z)
{
  return std::fma(x,y,z);
}

__device__

float __fmul_rd(float x, float y)
{
  return x*y;
}

__device__

float __fmul_rn(float x, float y)
{
  return x*y;
}

__device__

float __fmul_ru(float x, float y)
{
  return x*y;
}

__device__

float __fmul_rz(float x, float y)
{
  return x*y;
}

__device__

float __frcp_rd(float x)
{
  return 1.f/x;
}

__device__

float __frcp_rn(float x)
{
  return 1.f/x;
}

__device__

float __frcp_ru(float x)
{
  return 1.f/x;
}

__device__

float __frcp_rz(float x)
{
  return 1.f/x;
}

__device__

float __frsqrt_rn(float x)
{
  return 1.f/std::sqrt(x);
}

__device__

float __fsqrt_rd(float x)
{
  return std::sqrt(x);
}

__device__

float __fsqrt_rn(float x)
{
  return std::sqrt(x);
}

__device__

float __fsqrt_ru(float x)
{
  return std::sqrt(x);
}

__device__

float __fsqrt_rz(float x)
{
  return std::sqrt(x);
}

__device__

float __fsub_rd(float x, float y)
{
  return x-y;
}

__device__

float __fsub_rn(float x, float y)
{
  return x-y;
}

__device__

float __fsub_ru(float x, float y)
{
  return x-y;
}

__device__

float __fsub_rz(float x, float y)
{
  return x-y;
}

__device__

double __dadd_rd(double x, double y)
{
  return x+y;
}

__device__

double __dadd_rn(double x, double y)
{
  return x+y;
}

__device__

double __dadd_ru(double x, double y)
{
  return x+y;
}

__device__

double __dadd_rz(double x, double y)
{
  return x+y;
}

__device__

double __ddiv_rd(double x, double y)
{
  return x/y;
}

__device__

double __ddiv_rn(double x, double y)
{
  return x/y;
}

__device__

double __ddiv_ru(double x, double y)
{
  return x/y;
}

__device__

double __ddiv_rz(double x, double y)
{
  return x/y;
}

__device__

double __dmul_rd(double x, double y)
{
  return x*y;
}

__device__

double __dmul_rn(double x, double y)
{
  return x*y;
}

__device__

double __dmul_ru(double x, double y)
{
  return x*y;
}

__device__

double __dmul_rz(double x, double y)
{
  return x*y;
}

__device__

double __drcp_rd(double x)
{
  return 1./x;
}

__device__

double __drcp_rn(double x)
{
  return 1./x;
}

__device__

double __drcp_ru(double x)
{
  return 1./x;
}

__device__

double __drcp_rz(double x)
{
  return 1./x;
}

__device__

double __dsqrt_rd(double x)
{
  return std::sqrt(x);
}

__device__

double __dsqrt_rn(double x)
{
  return std::sqrt(x);
}

__device__

double __dsqrt_ru(double x)
{
  return std::sqrt(x);
}

__device__

double __dsqrt_rz(double x)
{
  return std::sqrt(x);
}

__device__

double __dsub_rd(double x, double y)
{
  return x - y;
}

__device__

double __dsub_rn(double x, double y)
{
  return x - y;
}

__device__

double __dsub_ru(double x, double y)
{
  return x - y;
}

__device__

double __dsub_rz(double x, double y)
{
  return x - y;
}

__device__

double __fma_rd(double x, double y, double z)
{
  return std::fma(x,y,z);
}

__device__

double __fma_rn(double x, double y, double z)
{
  return std::fma(x,y,z);
}

__device__

double __fma_ru(double x, double y, double z)
{
  return std::fma(x,y,z);
}

__device__

double __fma_rz(double x, double y, double z)
{
  return std::fma(x,y,z);
}


#endif // CPUCUDA_RUNTIME_H
