#include "backends.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#include "iree/modules/hal/types.h" // iree_hal_buffer_view_move_ref / _deref
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

} // namespace

struct IreeBackend : InferBackend
{
    iree_runtime_instance_t* instance = nullptr;
    iree_hal_device_t*       device   = nullptr;
    iree_runtime_session_t*  session  = nullptr;

    explicit IreeBackend(const std::string& vmfb_path, const char* driver)
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

    std::vector<float> infer(const Input& in) override
    {
        iree_vm_function_t fn;
        // IREE uses the JAX function name: "jit_inference.main"
        IREE_CHECK_OK(iree_runtime_session_lookup_function(session, iree_make_cstring_view("jit_inference.main"), &fn));

        // Build input list: [query, docs]
        iree_vm_list_t* inputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          /*initial_capacity=*/4,
                                          iree_allocator_system(),
                                          &inputs));

        iree_hal_dim_t query_shape[] = { (iree_hal_dim_t)in.query_dim };
        push_float_tensor(session, inputs, in.query.data(), query_shape, 1);

        iree_hal_dim_t docs_shape[] = { (iree_hal_dim_t)in.num_docs, (iree_hal_dim_t)in.doc_dim };
        push_float_tensor(session, inputs, in.docs.data(), docs_shape, 2);

        // Build (empty) output list
        iree_vm_list_t* outputs = nullptr;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                          /*initial_capacity=*/2,
                                          iree_allocator_system(),
                                          &outputs));

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
