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

#ifndef CPUCUDA_RUNTIME_HPP
#define CPUCUDA_RUNTIME_HPP

#include <thread>
#include <limits>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>

#include "queue.hpp"
#include "event.hpp"

#ifndef CPUCUDA_NO_OPENMP
#include <omp.h>
#endif

namespace cpucuda {
namespace detail {

using std::uintptr_t;

template<class Object>
class object_storage
{
public:

  // Need shared_ptr since e.g. hipEventDestroy()
  // is non-blocking and cleanup must only happen once
  // the event is complete
  using object_ptr = std::shared_ptr<Object>;

  struct item
  {
    std::uintptr_t id;
    object_ptr data;
  };

  uintptr_t store(object_ptr obj)
  {
    std::lock_guard<std::mutex> lock{_lock};

    for(std::uintptr_t i = 0; i < _data.size(); ++i)
      if(!_data[i])
      {
        _data[i] = std::move(obj);
        return i;
      }

    _data.push_back(obj);
    assert(_data.size() > 0);

    return _data.size()-1;
  }

  Object* get(uintptr_t id) const
  {
    std::lock_guard<std::mutex> lock{_lock};

    assert(this->is_valid(id));
    return _data[id].get();
  }

  object_ptr get_shared(uintptr_t id) const
  {
    std::lock_guard<std::mutex> lock{_lock};

    assert(this->is_valid(id));

    return _data[id];
  }

  void destroy(uintptr_t id)
  {
    std::lock_guard<std::mutex> lock{_lock};

    assert(this->is_valid(id));

    _data[id] = nullptr;
  }

  template<class Handler>
  void for_each(Handler h) const
  {
    std::lock_guard<std::mutex> lock{_lock};
    for(auto& obj : _data)
    {
      if(obj)
        h(obj.get());
    }
  }

  bool is_valid(uintptr_t id) const
  {
    if(id < 0 || id >= _data.size())
      return false;
    if(_data[id] == nullptr)
      return false;

    return true;
  }
private:
  mutable std::mutex _lock;
  std::vector<object_ptr> _data;
};

}

class stream
{
public:
  /// Construct stream - if master stream is not null,
  /// all operations are forwarded to the async queue
  /// of the master stream.
  /// This guarantees that operations are never overlapping
  /// with operations on the master stream (needed for default
  /// stream semantics)
  stream(stream* master_stream = nullptr)
  : _master_stream{master_stream}
  {
    if(!_master_stream)
      _queue = std::make_unique<detail::async_queue>();
  }

  template<class Func>
  void operator()(Func f)
  {
    this->execute(f);
  }

  void wait()
  {
    if(_master_stream)
      _master_stream->wait();
    else
      _queue->wait();
  }

  bool is_idle() const
  {
    if(_master_stream)
      return _master_stream->is_idle();

    return _queue->is_idle();
  }

private:

  template<class Func>
  void execute(Func f)
  {
    if(_master_stream)
      _master_stream->execute(f);
    else
      (*_queue)(f);
  }

  stream* _master_stream;
  std::unique_ptr<detail::async_queue> _queue;
};

typedef void (*__cpucuda_call_kernel_type)(dim3, dim3, dim3, void**, size_t);

class device
{
public:
	void submit_kernel(stream& execution_stream,
	                   dim3 grid_dim, dim3 block_dim,
	                   int shared_mem, const void *func, void **args)
  {
    execution_stream([=](){
      std::lock_guard<std::mutex> lock{this->_kernel_execution_mutex};
#ifndef CPUCUDA_NO_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
      for(unsigned g_z = 0; g_z < grid_dim.z; ++g_z) {
        //printf("g_z %u < %u\n", g_z, grid_dim.z);
        for(unsigned g_y = 0; g_y < grid_dim.y; ++g_y) {
          //printf("g_y %u < %u\n", g_y, grid_dim.y);
          for(unsigned g_x = 0; g_x < grid_dim.x; ++g_x) {
            dim3 block_idx = {g_x, g_y, g_z};
            //printf("block_idx %u\n", g_x + grid_dim.x * (g_y + grid_dim.y * g_z));
            ((__cpucuda_call_kernel_type) func)(grid_dim, block_idx, block_dim, args, shared_mem);
          }
        }
      }
      free(args);
    });
  }

  template<class Func>
  void submit_operation(stream& execution_stream, Func f)
  {
    execution_stream(f);
  }

  void barrier()
  {
#ifndef CPUCUDA_NO_OPENMP
    #pragma omp barrier
#endif
  }

  int get_max_threads()
  {
#ifndef CPUCUDA_NO_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }

  int get_num_compute_units()
  {
#ifndef CPUCUDA_NO_OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
  }

  std::size_t get_max_shared_memory() const
  { return std::numeric_limits<std::size_t>::max(); }

private:
  std::mutex _kernel_execution_mutex;
};


class runtime
{
  runtime()
  : _current_device{0}
  {
    _devices.push_back(std::make_unique<device>());
    // Create default stream
    int stream_id = _streams.store(std::make_unique<stream>());
#ifndef NDEBUG
    assert(stream_id == 0);
#else
    (void) stream_id;
#endif
  }
public:
  static runtime& get()
  {
    static runtime r;
    return r;
  }

  uintptr_t create_async_stream()
  {
    return _streams.store(std::make_unique<stream>());
  }

  uintptr_t create_blocking_stream()
  {
    return _streams.store(std::make_unique<stream>(_streams.get(0)));
  }

  void destroy_stream(uintptr_t stream_id)
  {
    assert(stream_id != 0);
    _streams.destroy(stream_id);
  }

  uintptr_t create_event()
  {
    return _events.store(std::make_unique<event>());
  }

  void destroy_event(uintptr_t event_id)
  {
    _events.destroy(event_id);
  }

  const detail::object_storage<stream>& streams() const
  {
    return _streams;
  }

  const detail::object_storage<event>& events() const
  {
    return _events;
  }

  device& dev() const noexcept
  {
    return *_devices[this->get_device()];
  }

  int get_num_devices() const noexcept
  {
    assert(_devices.size() == 1);
    return _devices.size();
  }

  int get_device() const noexcept
  {
    return _current_device;
  }

  void set_device(int device) noexcept
  {
    assert(device >= 0 && device < get_num_devices());
    _current_device = device;
  }


  template<class Func>
  void submit_operation(Func f, uintptr_t stream_id = 0)
  {
    auto s = this->_streams.get(stream_id);
    this->dev().submit_operation(*s, f);
  }

  template<class Func>
  void submit_kernel(dim3 grid, dim3 block,
                    int shared_mem, int stream, Func f)
  {
    auto s = this->_streams.get(stream);
    this->dev().submit_kernel(*s, grid, block, shared_mem, f);
  }

  template<class Func>
  void submit_unparallelized_kernel(int scratch_mem, int stream, Func f)
  {
    auto s = this->_streams.get(stream);
    this->dev().submit_kernel(*s, scratch_mem, f);
  }

	void submit_kernel(const void *func, dim3 grid, dim3 block, void **args,
	                   size_t shared_mem, uintptr_t stream)
	{
		auto s = this->_streams.get(stream);
		this->dev().submit_kernel(*s, grid, block, shared_mem, func, args);
		// If we are on the master stream, wait for execution to end TODO check how
		// this is achieve in the original hipCPU
		if (stream == 0)
			s->wait();
	}

private:
  mutable std::mutex _runtime_lock;

  detail::object_storage<stream> _streams;
  detail::object_storage<event> _events;
  // TODO: This should be thread local, but as long
  // as we are on the host system, we effectively
  // only have a single device
  int _current_device;

  std::vector<std::unique_ptr<device>> _devices;
};

}

#endif
