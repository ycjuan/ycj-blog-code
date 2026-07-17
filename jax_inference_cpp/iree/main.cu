#include <iostream>
#include <stdexcept>
#include <vector>

#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

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

static void push_float_tensor(iree_runtime_session_t* session,
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
                                               0,
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

int main()
{
    // query: [64], docs: [5, 128]
    const int query_dim = 64;
    const int num_docs  = 5;
    const int doc_dim   = 128;
    const int num_heads = 2;

    std::vector<float> query_data(query_dim, 0.5f);
    std::vector<float> docs_data(num_docs * doc_dim, 0.5f);

    iree_runtime_instance_options_t inst_opts;
    iree_runtime_instance_options_initialize(&inst_opts);
    iree_runtime_instance_options_use_all_available_drivers(&inst_opts);

    iree_runtime_instance_t* instance = nullptr;
    IREE_CHECK_OK(iree_runtime_instance_create(&inst_opts, iree_allocator_system(), &instance));

    iree_hal_device_t* device = nullptr;
    IREE_CHECK_OK(
        iree_runtime_instance_try_create_default_device(instance, iree_make_cstring_view("local-sync"), &device));

    iree_runtime_session_options_t sess_opts;
    iree_runtime_session_options_initialize(&sess_opts);

    iree_runtime_session_t* session = nullptr;
    IREE_CHECK_OK(
        iree_runtime_session_create_with_device(instance, &sess_opts, device, iree_allocator_system(), &session));

    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(session, "../model.vmfb"));

    iree_vm_function_t fn;
    IREE_CHECK_OK(iree_runtime_session_lookup_function(session, iree_make_cstring_view("jit_inference.main"), &fn));

    iree_vm_list_t* inputs = nullptr;
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                      /*initial_capacity=*/4,
                                      iree_allocator_system(),
                                      &inputs));

    iree_hal_dim_t query_shape[] = { (iree_hal_dim_t)query_dim };
    push_float_tensor(session, inputs, query_data.data(), query_shape, 1);

    iree_hal_dim_t docs_shape[] = { (iree_hal_dim_t)num_docs, (iree_hal_dim_t)doc_dim };
    push_float_tensor(session, inputs, docs_data.data(), docs_shape, 2);

    iree_vm_list_t* outputs = nullptr;
    IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                      /*initial_capacity=*/2,
                                      iree_allocator_system(),
                                      &outputs));

    IREE_CHECK_OK(iree_runtime_session_call(session, &fn, inputs, outputs));

    // Read output buffer view
    iree_vm_ref_t ref = iree_vm_ref_null();
    IREE_CHECK_OK(iree_vm_list_get_ref_assign(outputs, 0, &ref));
    iree_hal_buffer_view_t* out_bv = iree_hal_buffer_view_deref(ref);

    iree_hal_device_t* dev      = iree_runtime_session_device(session);
    iree_hal_buffer_t* buf      = iree_hal_buffer_view_buffer(out_bv);
    iree_device_size_t byte_len = (iree_device_size_t)(num_docs * num_heads) * sizeof(float);
    std::vector<float> scores(num_docs * num_heads);
    IREE_CHECK_OK(iree_hal_device_transfer_d2h(dev,
                                               buf,
                                               0,
                                               scores.data(),
                                               byte_len,
                                               IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                               iree_infinite_timeout()));

    std::cout << "Scores shape: [" << num_docs << ", " << num_heads << "]\n";
    std::cout << "Scores:\n";
    for (int d = 0; d < num_docs; ++d)
    {
        for (int h = 0; h < num_heads; ++h)
            std::cout << scores[d * num_heads + h] << " ";
        std::cout << "\n";
    }

    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
    iree_runtime_session_release(session);
    iree_hal_device_release(device);
    iree_runtime_instance_release(instance);
    return 0;
}
