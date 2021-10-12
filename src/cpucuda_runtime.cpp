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
#include <cstdint>

#include <cuda_runtime.h>

#include "detail/runtime.hpp"
#include "detail/malloc.hpp"

using std::uintptr_t;

#define _cpucuda_runtime (cpucuda::runtime::get())

// TODO this would have to be thread local
struct {
	dim3 gridDim, blockDim;
	size_t sharedMem;
	void *stream;
	bool used;
} __cpucudaCallConfiguration;

/*extern "C"*/ unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                    size_t sharedMem,
                                                    void *stream) {
	//assert(__cpucudaCallConfiguration.used);
	__cpucudaCallConfiguration.gridDim = gridDim;
	__cpucudaCallConfiguration.blockDim = blockDim;
	__cpucudaCallConfiguration.sharedMem = sharedMem;
	__cpucudaCallConfiguration.stream = stream;
	__cpucudaCallConfiguration.used = false;

	return 0;
}

cudaError_t __cpucudaLaunchKernel(
	const void* func,
	dim3 grid_dim,
	dim3 block_dim,
	void** args,
	size_t shared_mem,
	cudaStream_t stream)
{
	_cpucuda_runtime.submit_kernel(func, grid_dim, block_dim, args, shared_mem, stream);
	return cudaSuccess;
}

extern "C" cudaError_t __cpucudaLaunchKernelWithPushedConfiguration(
	const void* func,
	void** args)
{
	assert(__cpucudaCallConfiguration.used == false);
	dim3 grid_dim = __cpucudaCallConfiguration.gridDim;
	dim3 block_dim = __cpucudaCallConfiguration.blockDim;
	size_t shared_mem = __cpucudaCallConfiguration.sharedMem;
	cudaStream_t stream = (cudaStream_t) __cpucudaCallConfiguration.stream;
	__cpucudaCallConfiguration.used = true;

	return __cpucudaLaunchKernel(func, grid_dim, block_dim, args, shared_mem, stream);
}

cudaError_t cudaGetLastError(void)
{
	return cudaSuccess;
}

cudaError_t cudaMalloc(void** ptr, size_t size)
{
  *ptr = cpucuda::detail::aligned_malloc(cpucuda::detail::default_alignment, size);

  if(*ptr == nullptr)
    return cudaErrorMemoryAllocation;

  return cudaSuccess;
}

//cudaError_t cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
//cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent);


cudaError_t cudaFree(void* ptr)
{
  cpucuda::detail::aligned_free(ptr);
  return cudaSuccess;
}


cudaError_t cudaMallocHost(void** ptr, size_t size)
{
  return cudaMalloc(ptr, size);
}

cudaError_t cudaMallocManaged(void** ptr, size_t size, unsigned flags)
{
  return cudaMalloc(ptr, size);
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

  _cpucuda_runtime.set_device(device);
  return cudaSuccess;
}

//cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp_t* prop);

cudaError_t cudaStreamCreate(cudaStream_t* stream)
{
	*stream = reinterpret_cast<cudaStream_t>(_cpucuda_runtime.create_blocking_stream());
  return cudaSuccess;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
{
  if(flags == cudaStreamDefault)
    return cudaStreamCreate(stream);
  else if (flags == cudaStreamNonBlocking) 
  {
	  *stream = reinterpret_cast<cudaStream_t>(_cpucuda_runtime.create_async_stream());
    return cudaSuccess;
  }

  return cudaErrorInvalidValue;
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	_cpucuda_runtime.streams().get(reinterpret_cast<uintptr_t>(stream))->wait();
  return cudaSuccess;
}


cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	_cpucuda_runtime.destroy_stream(reinterpret_cast<uintptr_t>(stream));
  return cudaSuccess;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                            unsigned int flags)
{
  std::shared_ptr<cpucuda::event> evt = _cpucuda_runtime.events().get_shared(reinterpret_cast<uintptr_t>(event));
  _cpucuda_runtime.submit_operation([evt](){
    // TODO store error code
    evt->wait();
  }, reinterpret_cast<uintptr_t>(stream));
  return cudaSuccess;
}


cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	cpucuda::stream* s = _cpucuda_runtime.streams().get(reinterpret_cast<uintptr_t>(stream));
  
  if(s->is_idle())
    return cudaSuccess;

  return cudaErrorNotReady;
}

//TODO Make sure semantics are correct for all allowed values of flags

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                cudaStreamCallback_t callback, void *userData,
                                unsigned int flags) 
{
  _cpucuda_runtime.submit_operation([stream, callback, userData](){
    // TODO guarantee correct error propagation
    callback(stream, cudaSuccess, userData);
  }, reinterpret_cast<uintptr_t>(stream));
  return cudaSuccess;
}


cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                            cudaMemcpyKind copyKind, cudaStream_t stream)
{
	if(!_cpucuda_runtime.streams().is_valid(reinterpret_cast<uintptr_t>(stream)))
    return cudaErrorInvalidValue;

  _cpucuda_runtime.submit_operation([=](){
    memcpy(dst, src, sizeBytes);
  }, reinterpret_cast<uintptr_t>(stream));
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t sizeBytes,
                                   cudaMemcpyKind copyKind)
{
  cudaMemcpyAsync(dst, src, sizeBytes, copyKind, 0);
  _cpucuda_runtime.streams().get(0)->wait();
  return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  cudaMemcpyKind copyType,
                                  cudaStream_t stream)
{
  char* base_ptr = static_cast<char*>(const_cast<void*>(symbol));
  void* ptr = static_cast<void*>(base_ptr + offset);
  return cudaMemcpyAsync(ptr, src, sizeBytes, copyType, stream);
}


cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                    size_t sizeBytes, size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream)
{
  const void* ptr = 
    static_cast<const void*>(static_cast<const char*>(symbolName)+offset);
  return cudaMemcpyAsync(dst, ptr, sizeBytes, kind, stream);
}


cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                            size_t offset,
                            cudaMemcpyKind copyType)
{
  cudaError_t err = 
    cudaMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, copyType, 0);

  if(err != cudaSuccess)
    return err;

  _cpucuda_runtime.streams().get(0)->wait();
  return err;
}


cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbolName,
                               size_t sizeBytes, size_t offset,
                               cudaMemcpyKind kind) 
{
  cudaError_t err = 
    cudaMemcpyFromSymbolAsync(dst, symbolName, sizeBytes, offset, kind, 0);
    
  if(err != cudaSuccess)
    return err;

  _cpucuda_runtime.streams().get(0)->wait();
  return err;
}


cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);


cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, cudaMemcpyKind kind,
                                          cudaStream_t stream)
{
	if(!_cpucuda_runtime.streams().is_valid(reinterpret_cast<uintptr_t>(stream)))
    return cudaErrorInvalidValue;

  _cpucuda_runtime.submit_operation([=](){
    for(size_t row; row < height; ++row)
    {
      void* row_dst_begin = reinterpret_cast<char*>(dst) + row * dpitch;
      const void* row_src_begin = reinterpret_cast<const char*>(src) + row * spitch;

      memcpy(row_dst_begin, row_src_begin, width);
    }
  }, reinterpret_cast<uintptr_t>(stream));
  
  return cudaSuccess;
}


cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, cudaMemcpyKind kind)
{
  cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 0);

  if(err != cudaSuccess)
    return err;

  _cpucuda_runtime.streams().get(0)->wait();
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
  _cpucuda_runtime.streams().for_each([](cpucuda::stream* s){
    s->wait();
  });
  return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error)
{
	return "cudaGetErrorString not yet implemented...";
}

const char* cudaGetErrorName(cudaError_t error)
{
	return "cudaGetErrorName not yet implemented...";
}


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
                                        cudaStream_t stream)
{
	if(!_cpucuda_runtime.streams().is_valid(reinterpret_cast<uintptr_t>(stream)))
    return cudaErrorInvalidValue;
  
  _cpucuda_runtime.submit_operation([=](){
    memset(devPtr, value, count);
  }, reinterpret_cast<uintptr_t>(stream));

  return cudaSuccess;
}


cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
  cudaError_t err = cudaMemsetAsync(devPtr, value, count, 0);
  if(err != cudaSuccess)
    return err;

  _cpucuda_runtime.streams().get(0)->wait();
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p_prop, int device)
{
  if(device != 0)
    return cudaErrorInvalidDevice;

  static const char device_name[] = "cpucuda OpenMP host device";
  int max_dim = std::numeric_limits<int>::max();

  static_assert(sizeof device_name <= sizeof p_prop->name, "");
  memcpy(p_prop->name, device_name, sizeof device_name);

  // TODO: Find available memory
  p_prop->totalGlobalMem = std::numeric_limits<size_t>::max();
  p_prop->sharedMemPerBlock = _cpucuda_runtime.dev().get_max_shared_memory();
  p_prop->regsPerBlock = std::numeric_limits<int>::max();
  p_prop->warpSize = 1;
  p_prop->maxThreadsPerBlock = _cpucuda_runtime.dev().get_max_threads();
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
  p_prop->multiProcessorCount = _cpucuda_runtime.dev().get_num_compute_units();
  // TODO: Find actual value
  p_prop->l2CacheSize = std::numeric_limits<int>::max();
  p_prop->maxThreadsPerMultiProcessor = p_prop->maxThreadsPerBlock;
  p_prop->computeMode = 0;

  p_prop->concurrentKernels = 1;
  p_prop->pciBusID = 0;
  p_prop->pciDeviceID = 0;
  p_prop->sharedMemPerMultiprocessor = p_prop->sharedMemPerBlock;
  p_prop->isMultiGpuBoard = 0;
  p_prop->canMapHostMemory = 1;

  return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
{
  switch (attr) {
  case cudaDevAttrMaxBlockDimX: //2:
    *value = 1024; break;
  case cudaDevAttrMaxBlockDimY: //3:
    *value = 1024; break;
  case cudaDevAttrMaxBlockDimZ: //4:
    *value = 64; break;
  case cudaDevAttrMaxGridDimX: //5:
    *value = 2147483647; break;
  case cudaDevAttrMaxGridDimY: //6:
    *value = 65535; break;
  case cudaDevAttrMaxGridDimZ: //7:
    *value = 65535; break;
  case cudaDevAttrMaxSharedMemoryPerBlock: //8:
    *value = 49152; break;
  case cudaDevAttrWarpSize: //10:
    *value = 32; break;
  case cudaDevAttrMaxRegistersPerBlock: //12:
    *value = 65536; break;
  case cudaDevAttrTextureAlignment: //14:
    *value = 512; break;
  case cudaDevAttrMultiProcessorCount: //16:
    *value = 28; break;
  case cudaDevAttrPciDeviceId: //34:
    *value = 0; break;
  case cudaDevAttrTccDriver: //35:
    *value = 0; break;
  case cudaDevAttrMaxThreadsPerMultiProcessor: //39:
    *value = 2048; break;
  case cudaDevAttrComputeCapabilityMajor: //75:
    *value = 6; break;
  case cudaDevAttrComputeCapabilityMinor: //76:
    *value = 1; break;
  case cudaDevAttrMaxSharedMemoryPerMultiprocessor: //81:
    *value = 98304; break;
  case cudaDevAttrCooperativeLaunch: //95:
    *value = 1; break;
    // TODO look properly at below attrs
  case cudaDevAttrComputeMode: // 20:
    *value = 0; break; // cudaComputeModeDefault
  case cudaDevAttrClockRate: // 13:
    *value = 100; break;
  default:
    fprintf(stderr, "unknown attr %d", attr);
    return cudaErrorApiFailureBase;
  }
  return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                        const void* func,
                                                        int blockSize,
                                                        size_t dynamicSMemSize);

//cudaError_t cudaPointerGetAttributes(cudaPointerAttribute_t* attributes, void* ptr);

cudaError_t cudaMemGetInfo(size_t* free, size_t* total);


cudaError_t cudaEventCreate(cudaEvent_t* event)
{
	*event = reinterpret_cast<cudaEvent_t>(_cpucuda_runtime.create_event());
  return cudaSuccess;
}


cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  if(!_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(event)) ||
     !_cpucuda_runtime.streams().is_valid(reinterpret_cast<uintptr_t>(stream)))
    return cudaErrorInvalidValue;

  std::shared_ptr<cpucuda::event> evt = _cpucuda_runtime.events().get_shared(reinterpret_cast<uintptr_t>(event));
  _cpucuda_runtime.submit_operation([evt](){
    evt->mark_as_finished();
  }, reinterpret_cast<uintptr_t>(stream));
  return cudaSuccess;
}


cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
  if(!_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(event)))
    return cudaErrorInvalidValue;

  cpucuda::event* evt = _cpucuda_runtime.events().get(reinterpret_cast<uintptr_t>(event));
  evt->wait();

  if(evt->is_complete())
    return cudaSuccess;

  return cudaErrorUnknown;
}


cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop)
{
	if(!_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(start)) || !_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(stop)))
    return cudaErrorInvalidValue;

	cpucuda::event* start_evt = _cpucuda_runtime.events().get(reinterpret_cast<uintptr_t>(start));
	cpucuda::event* stop_evt = _cpucuda_runtime.events().get(reinterpret_cast<uintptr_t>(stop));
  if(start_evt->is_complete() && stop_evt->is_complete())
  {
    *ms = static_cast<float>(stop_evt->timestamp_ns() - start_evt->timestamp_ns()) / 1e6f;
    return cudaSuccess;
  }

  return cudaErrorUnknown;
}


cudaError_t cudaEventDestroy(cudaEvent_t event)
{
  if(!_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(event)))
    return cudaErrorInvalidValue;

  _cpucuda_runtime.destroy_event(reinterpret_cast<uintptr_t>(event));
  return cudaSuccess;
}

cudaError_t cudaDriverGetVersion(int* driverVersion);


cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)
{
  *runtimeVersion = 99999;
  return cudaSuccess;
}

/*
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
                                            cudaStream_t stream);

// Profile APIs:
cudaError_t cudaProfilerStart();
cudaError_t cudaProfilerStop();

cudaError_t cudaSetDeviceFlags(unsigned int flags);

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);

*/

cudaError_t cudaEventQuery(cudaEvent_t event)
{
	if(!_cpucuda_runtime.events().is_valid(reinterpret_cast<uintptr_t>(event)))
    return cudaErrorInvalidValue;

	bool is_ready = _cpucuda_runtime.events().get(reinterpret_cast<uintptr_t>(event))->is_complete();

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
                                                           size_t dynamicSMemSize,
                                                           int blockSizeLimit,
                                                           unsigned int flags);

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

/*

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
*/

#endif // CPUCUDA_RUNTIME_H
