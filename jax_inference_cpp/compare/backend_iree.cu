#include "backends.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

// Abort on non-OK IREE status; frees the status object.
#define IREE_CHECK_OK(expr)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        iree_status_t _s = (expr);                                                                                     \
        if (!iree_status_is_ok(_s))                                                                                    \
        {                                                                                                              \
            iree_status_fprint(stderr, _s);                                                                            \
            iree_status_free(_s);                                                                                      \
            throw std::runtime_error("IREE error: " #expr);                                                            \
        }                                                                                                              \
    } while (0)

namespace
{

// Create a HAL buffer view wrapping host memory and push it onto a VM list.
void push_float_tensor(iree_runtime_session_t* session,
                       iree_vm_list_t*         list,
                       const float*            data,
                       const iree_hal_dim_t*   shape,
                       iree_host_size_t        rank)
{
    iree_hal_device_t*    device    = iree_runtime_session_device(session);
    iree_hal_allocator_t* allocator = iree_runtime_session_device_allocator(session);

    iree_device_size_t byte_len = sizeof(float);
    for (iree_host_size_t i = 0; i < rank; ++i)
        byte_len *= (iree_device_size_t)shape[i];

    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        allocator,
        (iree_hal_buffer_params_t) { .usage  = IREE_HAL_BUFFER_USAGE_DEFAULT,
                                     .access = IREE_HAL_MEMORY_ACCESS_ALL,
                                     .type   = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE },
        byte_len,
        &buffer));

    IREE_CHECK_OK(iree_hal_device_transfer_h2d(device,
                                               data,
                                               buffer,
                                               /*target_offset=*/0,
                                               byte_len,
                                               IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                               iree_infinite_timeout()));

    iree_hal_buffer_view_t* bv = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_create(buffer,
                                              rank,
                                              shape,
                                              IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                              IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                              iree_allocator_system(),
                                              &bv));
    iree_hal_buffer_release(buffer);

    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(bv);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(list, &ref));
}

// Import an existing CUDA device pointer as an IREE buffer view and push it onto a VM list.
void push_device_tensor(iree_runtime_session_t* session,
                        iree_vm_list_t*         list,
                        const float*            d_ptr,
                        const iree_hal_dim_t*   shape,
                        iree_host_size_t        rank)
{
    iree_hal_allocator_t* allocator = iree_runtime_session_device_allocator(session);

    iree_device_size_t byte_len = sizeof(float);
    for (iree_host_size_t i = 0; i < rank; ++i)
        byte_len *= (iree_device_size_t)shape[i];

    iree_hal_external_buffer_t ext_buf   = {};
    ext_buf.type                         = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
    ext_buf.flags                        = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
    ext_buf.size                         = byte_len;
    ext_buf.handle.device_allocation.ptr = (uint64_t)(uintptr_t)d_ptr;

    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(
        iree_hal_allocator_import_buffer(allocator,
                                         (iree_hal_buffer_params_t) { .usage  = IREE_HAL_BUFFER_USAGE_DEFAULT,
                                                                      .access = IREE_HAL_MEMORY_ACCESS_READ,
                                                                      .type   = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL },
                                         &ext_buf,
                                         iree_hal_buffer_release_callback_null(),
                                         &buffer));

    iree_hal_buffer_view_t* bv = nullptr;
    IREE_CHECK_OK(iree_hal_buffer_view_create(buffer,
                                              rank,
                                              shape,
                                              IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                              IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                              iree_allocator_system(),
                                              &bv));
    iree_hal_buffer_release(buffer);

    iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(bv);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(list, &ref));
}

// Read float data out of the first element (a buffer_view) on a VM output list.
std::vector<float> read_float_tensor(iree_runtime_session_t* session, iree_vm_list_t* list, int n_elems)
{
    iree_hal_device_t* device = iree_runtime_session_device(session);

    iree_vm_ref_t ref = iree_vm_ref_null();
    IREE_CHECK_OK(iree_vm_list_get_ref_assign(list, 0, &ref));

    iree_hal_buffer_view_t* bv = iree_hal_buffer_view_deref(ref);
    if (!bv)
        throw std::runtime_error("IREE: output is not a buffer_view");

    iree_hal_buffer_t* buf      = iree_hal_buffer_view_buffer(bv);
    iree_device_size_t byte_len = (iree_device_size_t)n_elems * sizeof(float);

    std::vector<float> out(n_elems);
    IREE_CHECK_OK(iree_hal_device_transfer_d2h(device,
                                               buf,
                                               /*source_offset=*/0,
                                               out.data(),
                                               byte_len,
                                               IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                               iree_infinite_timeout()));
    return out;
}

// Copy float data from the first output buffer_view into a device pointer.
// IREE's CUDA output buffers are DEVICE_LOCAL (not host-mappable), so we
// transfer via a host staging buffer then cudaMemcpy H2D into d_dst.
void copy_output_to_device(iree_runtime_session_t* session, iree_vm_list_t* list, float* d_dst, int n_elems)
{
    auto host_buf = read_float_tensor(session, list, n_elems);
    cudaMemcpy(d_dst, host_buf.data(), (size_t)n_elems * sizeof(float), cudaMemcpyHostToDevice);
}

} // namespace

struct IreeBackend : InferBackend
{
    iree_runtime_instance_t* instance  = nullptr;
    iree_hal_device_t*       device    = nullptr;
    iree_runtime_session_t*  session   = nullptr;
    bool                     use_cuda_ = false;

    explicit IreeBackend(const std::string& vmfb_path, const char* driver)
        : use_cuda_(std::string(driver) == "cuda")
    {
        iree_runtime_instance_options_t inst_opts;
        iree_runtime_instance_options_initialize(&inst_opts);
        iree_runtime_instance_options_use_all_available_drivers(&inst_opts);
        IREE_CHECK_OK(iree_runtime_instance_create(&inst_opts, iree_allocator_system(), &instance));

        IREE_CHECK_OK(
            iree_runtime_instance_try_create_default_device(instance, iree_make_cstring_view(driver), &device));

        iree_runtime_session_options_t sess_opts;
        iree_runtime_session_options_initialize(&sess_opts);
        IREE_CHECK_OK(
            iree_runtime_session_create_with_device(instance, &sess_opts, device, iree_allocator_system(), &session));

        IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path.c_str()));
    }

    ~IreeBackend()
    {
        if (session)
            iree_runtime_session_release(session);
        if (device)
            iree_hal_device_release(device);
        if (instance)
            iree_runtime_instance_release(instance);
    }

    bool supports_device_infer() const override { return use_cuda_; }

    // Kernel-only: import device pointers directly, avoiding H2D copies for inputs.
    // Output is D2D-copied from IREE's internal buffer into d_scores.
    void infer_device(const float* d_query,
                      const float* d_docs,
                      float*       d_scores,
                      int          query_dim,
                      int          doc_dim,
                      int          num_docs,
                      int          num_heads) override
    {
        iree_vm_function_t fn;
        IREE_CHECK_OK(iree_runtime_session_lookup_function(session, iree_make_cstring_view("jit_inference.main"), &fn));

        iree_vm_list_t* inputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 4, iree_allocator_system(), &inputs));

        iree_hal_dim_t query_shape[] = { (iree_hal_dim_t)query_dim };
        push_device_tensor(session, inputs, d_query, query_shape, 1);

        iree_hal_dim_t docs_shape[] = { (iree_hal_dim_t)num_docs, (iree_hal_dim_t)doc_dim };
        push_device_tensor(session, inputs, d_docs, docs_shape, 2);

        iree_vm_list_t* outputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2, iree_allocator_system(), &outputs));

        IREE_CHECK_OK(iree_runtime_session_call(session, &fn, inputs, outputs));

        copy_output_to_device(session, outputs, d_scores, num_docs * num_heads);

        iree_vm_list_release(inputs);
        iree_vm_list_release(outputs);
    }

    std::vector<float> infer(const Input& in) override
    {
        if (use_cuda_)
        {
            float* d_query  = nullptr;
            float* d_docs   = nullptr;
            float* d_scores = nullptr;
            cudaMalloc(&d_query, in.query.size() * sizeof(float));
            cudaMalloc(&d_docs, in.docs.size() * sizeof(float));
            cudaMalloc(&d_scores, (size_t)in.num_docs * in.num_heads * sizeof(float));

            cudaMemcpy(d_query, in.query.data(), in.query.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_docs, in.docs.data(), in.docs.size() * sizeof(float), cudaMemcpyHostToDevice);

            infer_device(d_query, d_docs, d_scores, in.query_dim, in.doc_dim, in.num_docs, in.num_heads);

            std::vector<float> scores(in.num_docs * in.num_heads);
            cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_query);
            cudaFree(d_docs);
            cudaFree(d_scores);
            return scores;
        }

        iree_vm_function_t fn;
        IREE_CHECK_OK(iree_runtime_session_lookup_function(session, iree_make_cstring_view("jit_inference.main"), &fn));

        iree_vm_list_t* inputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 4, iree_allocator_system(), &inputs));

        iree_hal_dim_t query_shape[] = { (iree_hal_dim_t)in.query_dim };
        push_float_tensor(session, inputs, in.query.data(), query_shape, 1);

        iree_hal_dim_t docs_shape[] = { (iree_hal_dim_t)in.num_docs, (iree_hal_dim_t)in.doc_dim };
        push_float_tensor(session, inputs, in.docs.data(), docs_shape, 2);

        iree_vm_list_t* outputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2, iree_allocator_system(), &outputs));

        IREE_CHECK_OK(iree_runtime_session_call(session, &fn, inputs, outputs));

        auto scores = read_float_tensor(session, outputs, in.num_docs * in.num_heads);

        iree_vm_list_release(inputs);
        iree_vm_list_release(outputs);
        return scores;
    }
};

std::unique_ptr<InferBackend> make_iree(const Paths& paths, const Input& /*shape_hint*/)
{
    return std::make_unique<IreeBackend>(paths.iree_model, "local-sync");
}

std::unique_ptr<InferBackend> make_iree_cuda(const Paths& paths, const Input& /*shape_hint*/)
{
    return std::make_unique<IreeBackend>(paths.iree_cuda_model, "cuda");
}
