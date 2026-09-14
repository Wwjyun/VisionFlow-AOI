#define VISIONFLOW_CUDA_EXPORTS
#include "visionflow_cuda.h"
#include "visionflow_cuda_internal.cuh"
// The exact-median export radix-sorts its order keys with the CCCL CUB device algorithm that ships
// with the CUDA Toolkit. CUB refuses to compile under the traditional MSVC preprocessor, so
// gpu/cuda_project.json declares /Zc:preprocessor for both native targets and
// gpu/build_cuda_dll.ps1 passes it through -Xcompiler.
#include <cub/device/device_radix_sort.cuh>
// which gpu/cuda_project.json now opts into for the whole project.
#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int SCAN_THREADS = 256;
constexpr int TRANSPOSE_TILE = 32;
constexpr int TRANSPOSE_ROWS = 8;
constexpr int MAX_GAUSSIAN_KERNEL = 127;
constexpr int TIMING_EVENT_COUNT = 12;
enum TimingEventIndex {
    TIMING_START = 0,
    TIMING_AFTER_INPUT = 1,
    TIMING_AFTER_KERNEL = 2,
    TIMING_AFTER_OUTPUT = 3,
    TIMING_GAUSSIAN_START = 4,
    TIMING_GAUSSIAN_END = 5,
    TIMING_ADAPTIVE_START = 6,
    TIMING_ADAPTIVE_END = 7,
    TIMING_THRESHOLD_START = 8,
    TIMING_THRESHOLD_END = 9,
    TIMING_MORPHOLOGY_START = 10,
    TIMING_MORPHOLOGY_END = 11,
};

constexpr unsigned int GAUSSIAN_FIXED_SHIFT = 8;
constexpr unsigned int GAUSSIAN_FIXED_SCALE = 1U << GAUSSIAN_FIXED_SHIFT;
constexpr unsigned int GAUSSIAN_FINAL_ROUND =
    1U << (GAUSSIAN_FIXED_SHIFT * 2 - 1);
__constant__ uint16_t gaussian_weights[MAX_GAUSSIAN_KERNEL];
// float32 coefficients for the optional vf_gaussian_blur_f32 export. Separate symbol from the
// uint16 fixed-point table above because that one serves the u8 export's rounding contract.
__constant__ float gaussian_f32_weights[MAX_GAUSSIAN_KERNEL];
// Odd kernel sizes the float32 export has been verified against cv2.GaussianBlur. The verified
// range is every odd size in [3, MAX_GAUSSIAN_KERNEL]; anything else is reported as
// VF_CUDA_UNSUPPORTED instead of being computed unvalidated. Evidence:
// outputs_validation/cnr_profile/gaussian_f32_equivalence.txt (tools/gaussian_f32_equivalence.py).
constexpr int GAUSSIAN_F32_MIN_KERNEL = 3;

// Template Anchor Grid scratch planes owned by the persistent context. Declared here because the
// context struct sizes its plane arrays from them.
constexpr int MATCH_SUM_PREFIX_PLANE = 0;      // int64: vertical prefix of the horizontal window sum
constexpr int MATCH_SQUARE_PREFIX_PLANE = 1;   // int64: vertical prefix of its sum of squares
constexpr int MATCH_PLANE_COUNT = 2;

struct PersistentContext {
    uint8_t* u8[5]{};
    size_t u8_capacity[5]{};
    uint32_t* gaussian_buffer = nullptr;
    size_t gaussian_capacity = 0;
    unsigned long long* u64[2]{};
    size_t u64_capacity[2]{};
    std::vector<uint8_t*> dag_u8;
    std::vector<size_t> dag_u8_capacity;
    uint8_t* resident_u8 = nullptr;
    size_t resident_capacity = 0;
    int resident_width = 0;
    int resident_height = 0;
    int resident_channels = 0;
    uint64_t resident_generation = 0;
    // Template Anchor Grid scratch: three grow-only int64 planes of output_width x output_height,
    // one int64 plane of template column sums, and the candidate/result slots. Kept separate from
    // u64[] because those are plan scratch and reserve_device only grows.
    long long* match_plane[MATCH_PLANE_COUNT]{};
    size_t match_plane_capacity[MATCH_PLANE_COUNT]{};
    long long* match_candidates = nullptr;
    size_t match_candidate_capacity = 0;
    int match_candidate_output_width = 0;
    // Contour extension scratch: the padded label image, the discovery-order result, and the
    // OpenCV-order result the download export copies out. All grow-only.
    signed char* contour_label = nullptr;
    size_t contour_label_capacity = 0;
    int32_t* contour_offsets = nullptr;
    size_t contour_offset_capacity = 0;
    int32_t* contour_points = nullptr;
    size_t contour_point_capacity = 0;
    int32_t* contour_out_offsets = nullptr;
    size_t contour_out_offset_capacity = 0;
    int32_t* contour_out_points = nullptr;
    size_t contour_out_point_capacity = 0;
    int* contour_counts = nullptr;
    size_t contour_count_capacity = 0;
    // RETR_LIST transition list: one exact row count, the row segment table, and the raster-ordered
    // padded label indices the fast scan walks.
    int* contour_row_counts = nullptr;
    size_t contour_row_count_capacity = 0;
    int* contour_row_start = nullptr;
    size_t contour_row_start_capacity = 0;
    int32_t* contour_transitions = nullptr;
    size_t contour_transition_capacity = 0;
    int contour_count = 0;
    int contour_point_count = 0;
    uint64_t contour_generation = 0;
    bool contour_result_valid = false;
    // Exact-median scratch: the uploaded float values, their monotone-orderable uint32 order keys,
    // the radix-sorted keys, the cub temporary storage and a one-word NaN-presence flag. All
    // grow-only and deliberately separate from u8[]/u64[], which are plan scratch.
    float* median_values = nullptr;
    size_t median_value_capacity = 0;
    uint32_t* median_keys = nullptr;
    size_t median_key_capacity = 0;
    uint32_t* median_sorted_keys = nullptr;
    size_t median_sorted_key_capacity = 0;
    uint8_t* median_sort_scratch = nullptr;
    size_t median_sort_scratch_capacity = 0;
    int* median_nan_flag = nullptr;
    size_t median_nan_flag_capacity = 0;
    // float32 Gaussian scratch for the optional vf_gaussian_blur_f32 export: the uploaded source
    // rectangle, the horizontal intermediate and the packed result that is copied back. Grow-only
    // and deliberately separate from u8[]/u64[], which are plan scratch that reserve_device only
    // grows, so sharing them would mix a plan's buffer contents with this operator's input.
    float* gaussian_f32_input = nullptr;
    size_t gaussian_f32_input_capacity = 0;
    float* gaussian_f32_intermediate = nullptr;
    size_t gaussian_f32_intermediate_capacity = 0;
    float* gaussian_f32_output = nullptr;
    size_t gaussian_f32_output_capacity = 0;
    unsigned long long allocation_count = 0;
    cudaStream_t stream = nullptr;
    cudaError_t initialization_error = cudaSuccess;
    cudaEvent_t timing_events[TIMING_EVENT_COUNT]{};
    VfCudaTimingsV1 last_timings{};
    float pending_allocation_ms = 0.0f;
    bool timing_input_is_host = true;
    bool timing_has_gaussian = false;
    bool timing_has_adaptive = false;
    bool timing_has_threshold = false;
    bool timing_has_morphology = false;

    PersistentContext() {
        initialization_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        last_timings.struct_size = sizeof(VfCudaTimingsV1);
        last_timings.version = 1;
        if (initialization_error == cudaSuccess) {
            for (cudaEvent_t& event : timing_events) {
                initialization_error = cudaEventCreate(&event);
                if (initialization_error != cudaSuccess) break;
            }
        }
    }

    ~PersistentContext() {
        for (void* pointer : u8) visionflow_cuda::free_device(pointer);
        visionflow_cuda::free_device(gaussian_buffer);
        for (void* pointer : u64) visionflow_cuda::free_device(pointer);
        for (void* pointer : dag_u8) visionflow_cuda::free_device(pointer);
        for (long long* plane : match_plane) visionflow_cuda::free_device(plane);
        visionflow_cuda::free_device(match_candidates);
        visionflow_cuda::free_device(contour_label);
        visionflow_cuda::free_device(contour_offsets);
        visionflow_cuda::free_device(contour_points);
        visionflow_cuda::free_device(contour_out_offsets);
        visionflow_cuda::free_device(contour_out_points);
        visionflow_cuda::free_device(contour_counts);
        visionflow_cuda::free_device(contour_row_counts);
        visionflow_cuda::free_device(contour_row_start);
        visionflow_cuda::free_device(contour_transitions);
        visionflow_cuda::free_device(median_values);
        visionflow_cuda::free_device(median_keys);
        visionflow_cuda::free_device(median_sorted_keys);
        visionflow_cuda::free_device(median_sort_scratch);
        visionflow_cuda::free_device(median_nan_flag);
        visionflow_cuda::free_device(gaussian_f32_input);
        visionflow_cuda::free_device(gaussian_f32_intermediate);
        visionflow_cuda::free_device(gaussian_f32_output);
        visionflow_cuda::free_device(resident_u8);
        for (cudaEvent_t event : timing_events) {
            if (event != nullptr) cudaEventDestroy(event);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

float elapsed_host_ms(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

void reset_timing(PersistentContext* context, bool input_is_host) {
    float context_create_ms = context->last_timings.context_create_ms;
    float allocation_ms = context->pending_allocation_ms;
    context->pending_allocation_ms = 0.0f;
    context->last_timings = {};
    context->last_timings.struct_size = sizeof(VfCudaTimingsV1);
    context->last_timings.version = 1;
    context->last_timings.context_create_ms = context_create_ms;
    context->last_timings.allocation_ms = allocation_ms;
    context->timing_input_is_host = input_is_host;
    context->timing_has_gaussian = false;
    context->timing_has_adaptive = false;
    context->timing_has_threshold = false;
    context->timing_has_morphology = false;
    cudaEventRecord(context->timing_events[TIMING_START], context->stream);
}

void finalize_timing(PersistentContext* context) {
    float input_ms = 0.0f;
    cudaEventElapsedTime(
        &input_ms, context->timing_events[TIMING_START],
        context->timing_events[TIMING_AFTER_INPUT]);
    if (context->timing_input_is_host) context->last_timings.h2d_ms = input_ms;
    else context->last_timings.device_copy_ms = input_ms;
    cudaEventElapsedTime(
        &context->last_timings.kernel_ms,
        context->timing_events[TIMING_AFTER_INPUT],
        context->timing_events[TIMING_AFTER_KERNEL]);
    cudaEventElapsedTime(
        &context->last_timings.d2h_ms,
        context->timing_events[TIMING_AFTER_KERNEL],
        context->timing_events[TIMING_AFTER_OUTPUT]);
    cudaEventElapsedTime(
        &context->last_timings.total_device_ms,
        context->timing_events[TIMING_START],
        context->timing_events[TIMING_AFTER_OUTPUT]);
    if (context->timing_has_gaussian) {
        cudaEventElapsedTime(
            &context->last_timings.gaussian_ms,
            context->timing_events[TIMING_GAUSSIAN_START],
            context->timing_events[TIMING_GAUSSIAN_END]);
    }
    if (context->timing_has_adaptive) {
        cudaEventElapsedTime(
            &context->last_timings.adaptive_integral_ms,
            context->timing_events[TIMING_ADAPTIVE_START],
            context->timing_events[TIMING_ADAPTIVE_END]);
    }
    if (context->timing_has_threshold) {
        cudaEventElapsedTime(
            &context->last_timings.threshold_ms,
            context->timing_events[TIMING_THRESHOLD_START],
            context->timing_events[TIMING_THRESHOLD_END]);
    }
    if (context->timing_has_morphology) {
        cudaEventElapsedTime(
            &context->last_timings.morphology_ms,
            context->timing_events[TIMING_MORPHOLOGY_START],
            context->timing_events[TIMING_MORPHOLOGY_END]);
    }
}

enum AreaResizeMode {
    AREA_RESIZE_COPY = 0,
    AREA_RESIZE_FAST_2X2 = 1,
    AREA_RESIZE_FAST_INTEGER = 2,
    AREA_RESIZE_GENERAL = 3,
};

// Device tables reproducing OpenCV INTER_AREA downscale for one source/target shape.
// indices: [x offsets (dw+1)] [y offsets (dh+1)] [x sources] [y sources]
// alphas:  [x weights] [y weights]
struct AreaResizeTables {
    int mode = AREA_RESIZE_COPY;
    int scale_x = 1;
    int scale_y = 1;
    float inverse_area = 1.0f;
    int x_entries = 0;
    int* indices = nullptr;
    float* alphas = nullptr;

    AreaResizeTables() = default;
    AreaResizeTables(const AreaResizeTables&) = delete;
    AreaResizeTables& operator=(const AreaResizeTables&) = delete;
    ~AreaResizeTables() {
        visionflow_cuda::free_device(indices);
        visionflow_cuda::free_device(alphas);
    }
};

struct NativePlan {
    PersistentContext* context = nullptr;
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
    int input_channels = 0;
    int output_channels = 0;
    std::vector<VfPlanOperatorV1> operators;
    std::vector<std::unique_ptr<AreaResizeTables>> area_resizes;
};

struct NativeDagPlan {
    PersistentContext* context = nullptr;
    int width = 0;
    int height = 0;
    int input_channels = 0;
    std::vector<VfPlanOperatorV1> operators;
    std::vector<int> node_channels;
    std::vector<int> output_nodes;
};

struct NativeRoiBatch {
    PersistentContext* context = nullptr;
    uint8_t* data = nullptr;
    VfRoiV1* device_rois = nullptr;
    int count = 0;
    int width = 0;
    int height = 0;
    int channels = 0;

    ~NativeRoiBatch() {
        visionflow_cuda::free_device(data);
        visionflow_cuda::free_device(device_rois);
    }
};

template <typename T>
int reserve_device(
    T** pointer,
    size_t* capacity,
    size_t count,
    unsigned long long* allocation_count = nullptr) {
    if (pointer == nullptr || capacity == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (*pointer != nullptr && *capacity >= count) return VF_CUDA_OK;
    T* replacement = nullptr;
    cudaError_t error = cudaMalloc(&replacement, count * sizeof(T));
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    visionflow_cuda::free_device(*pointer);
    *pointer = replacement;
    *capacity = count;
    if (allocation_count != nullptr) ++(*allocation_count);
    return VF_CUDA_OK;
}

template <typename T>
int reserve_exact(
    T** pointer,
    size_t* capacity,
    size_t count,
    unsigned long long* allocation_count = nullptr) {
    if (pointer == nullptr || capacity == nullptr || count == 0 || count > SIZE_MAX / sizeof(T)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (*pointer != nullptr && *capacity == count) return VF_CUDA_OK;
    T* replacement = nullptr;
    cudaError_t error = cudaMalloc(&replacement, count * sizeof(T));
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    visionflow_cuda::free_device(*pointer);
    *pointer = replacement;
    *capacity = count;
    if (allocation_count != nullptr) ++(*allocation_count);
    return VF_CUDA_OK;
}

int prepare_gaussian_weights(
    int kernel,
    int* radius_out,
    cudaStream_t stream = nullptr) {
    if (radius_out == nullptr || kernel < 3 || kernel % 2 == 0 || kernel > MAX_GAUSSIAN_KERNEL) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    std::vector<uint16_t> weights(kernel);
    int radius = kernel / 2;
    if (kernel == 3) {
        weights = {64, 128, 64};
    } else if (kernel == 5) {
        weights = {16, 64, 96, 64, 16};
    } else if (kernel == 7) {
        weights = {8, 28, 56, 72, 56, 28, 8};
    } else if (kernel == 9) {
        weights = {4, 13, 30, 51, 60, 51, 30, 13, 4};
    } else {
        double sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8;
        std::vector<double> normalized(kernel);
        double total = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            double value = std::exp(
                -(static_cast<double>(i) * i) / (2.0 * sigma * sigma));
            normalized[i + radius] = value;
            total += value;
        }
        for (double& value : normalized) value /= total;

        double error = 0.0;
        unsigned int side_sum = 0;
        for (int index = 0; index < radius; ++index) {
            double adjusted = normalized[index] * GAUSSIAN_FIXED_SCALE + error;
            unsigned int value = static_cast<unsigned int>(std::nearbyint(adjusted));
            error = adjusted - value;
            weights[index] = static_cast<uint16_t>(value);
            weights[kernel - 1 - index] = static_cast<uint16_t>(value);
            side_sum += value;
        }
        weights[radius] = static_cast<uint16_t>(GAUSSIAN_FIXED_SCALE - side_sum * 2);
    }
    const size_t weight_bytes = static_cast<size_t>(kernel) * sizeof(uint16_t);
    cudaError_t error = stream == nullptr
        ? cudaMemcpyToSymbol(gaussian_weights, weights.data(), weight_bytes)
        : cudaMemcpyToSymbolAsync(
            gaussian_weights,
            weights.data(),
            weight_bytes,
            0,
            cudaMemcpyHostToDevice,
            stream);
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    *radius_out = radius;
    return VF_CUDA_OK;
}

// Exact cv::getGaussianKernel(ksize, 0.0, CV_32F) coefficients for the float32 export.
//
// OpenCV derives sigma as 0.3*((ksize-1)*0.5-1)+0.8 only for ksize > 9; ksize 1, 3, 5, 7 and 9
// come from its fixed small-kernel table (SMALL_GAUSSIAN_SIZE is 9 in OpenCV 5.x, and the table is
// selected whenever sigma <= 0). Every other size is exp/sum/normalize in float64 followed by a
// single cast to float32. Both branches were compared bit-for-bit against
// cv2.getGaussianKernel(ksize, 0, cv2.CV_32F) for every odd ksize in [3, 127] by
// tools/gaussian_f32_equivalence.py, so the coefficients - and therefore the kernel sums - are
// identical to OpenCV's; only the accumulation order of the convolution differs.
bool gaussian_f32_kernel_supported(int kernel) {
    return kernel >= GAUSSIAN_F32_MIN_KERNEL && kernel <= MAX_GAUSSIAN_KERNEL && kernel % 2 == 1;
}

int prepare_gaussian_f32_weights(int kernel, int* radius_out, cudaStream_t stream = nullptr) {
    if (radius_out == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    if (!gaussian_f32_kernel_supported(kernel)) return VF_CUDA_UNSUPPORTED;
    // Rows are indexed by ksize / 2: 1, 3, 5, 7 and 9. The values are the exact float32 constants
    // OpenCV stores in small_gaussian_tab, already summing to 1.
    static const float small_kernels[5][9] = {
        {1.0f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f},
        {0.015625f, 0.05078125f, 0.1171875f, 0.19921875f, 0.234375f,
         0.19921875f, 0.1171875f, 0.05078125f, 0.015625f},
    };
    std::vector<float> values(static_cast<size_t>(kernel));
    if (kernel <= 9) {
        const float* fixed = small_kernels[kernel / 2];
        for (int i = 0; i < kernel; ++i) values[static_cast<size_t>(i)] = fixed[i];
    } else {
        const double sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8;
        const double scale = -0.5 / (sigma * sigma);
        std::vector<double> raw(static_cast<size_t>(kernel));
        double total = 0.0;
        for (int i = 0; i < kernel; ++i) {
            const double offset = static_cast<double>(i) - (kernel - 1) * 0.5;
            raw[static_cast<size_t>(i)] = std::exp(scale * offset * offset);
            total += raw[static_cast<size_t>(i)];
        }
        const double inverse = 1.0 / total;
        for (int i = 0; i < kernel; ++i) {
            values[static_cast<size_t>(i)] =
                static_cast<float>(raw[static_cast<size_t>(i)] * inverse);
        }
    }
    const size_t weight_bytes = static_cast<size_t>(kernel) * sizeof(float);
    cudaError_t error = stream == nullptr
        ? cudaMemcpyToSymbol(gaussian_f32_weights, values.data(), weight_bytes)
        : cudaMemcpyToSymbolAsync(
            gaussian_f32_weights,
            values.data(),
            weight_bytes,
            0,
            cudaMemcpyHostToDevice,
            stream);
    if (error != cudaSuccess) return visionflow_cuda::runtime_error(error);
    *radius_out = kernel / 2;
    return VF_CUDA_OK;
}

int adaptive_layout(
    int width,
    int height,
    int block,
    int* radius_out,
    int* padded_width_out,
    int* padded_height_out,
    size_t* padded_count_out) {
    if (width <= 0 || height <= 0 || block < 3 || block % 2 == 0 || radius_out == nullptr ||
        padded_width_out == nullptr || padded_height_out == nullptr || padded_count_out == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int radius = block / 2;
    if (radius > (INT_MAX - width) / 2 || radius > (INT_MAX - height) / 2) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int padded_width = width + radius * 2;
    int padded_height = height + radius * 2;
    if (static_cast<size_t>(padded_width) > SIZE_MAX / static_cast<size_t>(padded_height)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t padded_count = static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height);
    if (padded_count > SIZE_MAX / sizeof(unsigned long long)) return VF_CUDA_INVALID_ARGUMENT;
    *radius_out = radius;
    *padded_width_out = padded_width;
    *padded_height_out = padded_height;
    *padded_count_out = padded_count;
    return VF_CUDA_OK;
}

void write_reason(char* reason, int capacity, const char* message) {
    if (reason != nullptr && capacity > 0) strncpy_s(reason, capacity, message, _TRUNCATE);
}

int validate_plan_desc(
    const VfPlanDescV1* desc,
    int width,
    int height,
    int* output_channels,
    int* output_width,
    int* output_height,
    char* reason,
    int reason_capacity) {
    if (desc == nullptr || desc->struct_size != sizeof(VfPlanDescV1) ||
        desc->version != VF_CUDA_PLAN_VERSION || width <= 0 || height <= 0 ||
        (desc->input_channels != 1 && desc->input_channels != 3) ||
        desc->operator_count <= 0 || desc->operator_count > 64 || desc->operators == nullptr) {
        write_reason(reason, reason_capacity, "Invalid plan descriptor, version, shape or input channels");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) > SIZE_MAX / static_cast<size_t>(height) ||
        static_cast<size_t>(width) * static_cast<size_t>(height) >
            SIZE_MAX / static_cast<size_t>(desc->input_channels)) {
        write_reason(reason, reason_capacity, "Plan image shape overflows addressable memory");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) * static_cast<size_t>(height) > INT_MAX) {
        write_reason(reason, reason_capacity, "Plan image contains too many pixels for ABI v1 indexing");
        return VF_CUDA_UNSUPPORTED;
    }

    int channels = desc->input_channels;
    int current_width = width;
    int current_height = height;
    int previous_node = VF_PLAN_INPUT_NODE;
    for (int index = 0; index < desc->operator_count; ++index) {
        const VfPlanOperatorV1& op = desc->operators[index];
        if (op.struct_size != sizeof(VfPlanOperatorV1) || op.input_node != previous_node ||
            op.output_node <= previous_node) {
            write_reason(reason, reason_capacity, "Plan nodes must form one validated linear chain");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        switch (op.kind) {
            case VF_PLAN_GRAY:
                channels = 1;
                break;
            case VF_PLAN_GAUSSIAN:
                if (op.int_params[0] < 3 || op.int_params[0] % 2 == 0 ||
                    op.int_params[0] > MAX_GAUSSIAN_KERNEL) {
                    write_reason(reason, reason_capacity, "Gaussian kernel is unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_THRESHOLD:
                if (channels != 1 || op.int_params[0] < 0 || op.int_params[0] > 255 ||
                    op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1)) {
                    write_reason(reason, reason_capacity, "Threshold requires one channel and valid uint8 parameters");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                if (channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(current_width, current_height, op.int_params[0], &radius, &padded_width,
                                    &padded_height, &padded_count) != VF_CUDA_OK) {
                    write_reason(reason, reason_capacity, "AdaptiveMean shape or parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            }
            case VF_PLAN_MORPHOLOGY:
                if (op.int_params[0] < VF_MORPH_OPEN || op.int_params[0] > VF_MORPH_ERODE ||
                    op.int_params[1] < 3 || op.int_params[1] % 2 == 0 ||
                    op.int_params[2] < 1 || op.int_params[2] > INT_MAX / 2) {
                    write_reason(reason, reason_capacity, "Morphology parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_RESIZE_AREA:
                if (channels != 1 || op.int_params[0] <= 0 || op.int_params[1] <= 0 ||
                    op.int_params[0] > current_width || op.int_params[1] > current_height) {
                    write_reason(
                        reason, reason_capacity,
                        "Resize(area) requires one channel and non-expanding target dimensions");
                    return VF_CUDA_UNSUPPORTED;
                }
                current_width = op.int_params[0];
                current_height = op.int_params[1];
                break;
            default:
                write_reason(reason, reason_capacity, "Plan contains an unsupported operator kind");
                return VF_CUDA_UNSUPPORTED;
        }
        previous_node = op.output_node;
    }
    if (desc->output_node != previous_node) {
        write_reason(reason, reason_capacity, "Plan output node does not match the final operator");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (output_channels != nullptr) *output_channels = channels;
    if (output_width != nullptr) *output_width = current_width;
    if (output_height != nullptr) *output_height = current_height;
    write_reason(reason, reason_capacity, "Supported generic native linear plan");
    return VF_CUDA_OK;
}

int validate_dag_plan_desc(
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    std::vector<int>* node_channels,
    char* reason,
    int reason_capacity) {
    if (desc == nullptr || desc->struct_size != sizeof(VfDagPlanDescV1) ||
        desc->version != VF_CUDA_PLAN_VERSION || width <= 0 || height <= 0 ||
        (desc->input_channels != 1 && desc->input_channels != 3) ||
        desc->operator_count <= 0 || desc->operator_count > 64 || desc->operators == nullptr ||
        desc->output_count <= 0 || desc->output_count > desc->operator_count ||
        desc->output_nodes == nullptr) {
        write_reason(reason, reason_capacity, "Invalid DAG descriptor, version, shape or counts");
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(width) > SIZE_MAX / static_cast<size_t>(height) ||
        static_cast<size_t>(width) * static_cast<size_t>(height) > INT_MAX) {
        write_reason(reason, reason_capacity, "DAG image shape is unsupported");
        return VF_CUDA_UNSUPPORTED;
    }
    std::vector<int> channels;
    try {
        channels.resize(desc->operator_count);
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
    for (int index = 0; index < desc->operator_count; ++index) {
        const VfPlanOperatorV1& op = desc->operators[index];
        if (op.struct_size != sizeof(VfPlanOperatorV1) || op.output_node != index ||
            op.input_node < VF_PLAN_INPUT_NODE || op.input_node >= index) {
            write_reason(reason, reason_capacity, "DAG nodes must be topologically ordered");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        int input_channels = op.input_node == VF_PLAN_INPUT_NODE
            ? desc->input_channels : channels[op.input_node];
        int output_channels = input_channels;
        switch (op.kind) {
            case VF_PLAN_GRAY:
                output_channels = 1;
                break;
            case VF_PLAN_GAUSSIAN:
                if (op.int_params[0] < 3 || op.int_params[0] % 2 == 0 ||
                    op.int_params[0] > MAX_GAUSSIAN_KERNEL) {
                    write_reason(reason, reason_capacity, "DAG Gaussian kernel is unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_THRESHOLD:
                if (input_channels != 1 || op.int_params[0] < 0 || op.int_params[0] > 255 ||
                    op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1)) {
                    write_reason(reason, reason_capacity, "DAG Threshold parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                if (input_channels != 1 || op.int_params[1] < 0 || op.int_params[1] > 255 ||
                    (op.int_params[2] != 0 && op.int_params[2] != 1) ||
                    !std::isfinite(op.float_params[0]) ||
                    adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                    &padded_height, &padded_count) != VF_CUDA_OK) {
                    write_reason(reason, reason_capacity, "DAG AdaptiveMean parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            }
            case VF_PLAN_MORPHOLOGY:
                if (op.int_params[0] < VF_MORPH_OPEN || op.int_params[0] > VF_MORPH_ERODE ||
                    op.int_params[1] < 3 || op.int_params[1] % 2 == 0 ||
                    op.int_params[2] < 1 || op.int_params[2] > INT_MAX / 2) {
                    write_reason(reason, reason_capacity, "DAG Morphology parameters are unsupported");
                    return VF_CUDA_UNSUPPORTED;
                }
                break;
            default:
                write_reason(reason, reason_capacity, "DAG contains an unsupported operator kind");
                return VF_CUDA_UNSUPPORTED;
        }
        channels[index] = output_channels;
    }
    std::vector<bool> seen(desc->operator_count, false);
    for (int index = 0; index < desc->output_count; ++index) {
        int node = desc->output_nodes[index];
        if (node < 0 || node >= desc->operator_count || seen[node]) {
            write_reason(reason, reason_capacity, "DAG outputs must be unique existing nodes");
            return VF_CUDA_INVALID_ARGUMENT;
        }
        seen[node] = true;
    }
    if (node_channels != nullptr) *node_channels = std::move(channels);
    write_reason(reason, reason_capacity, "Supported generic native DAG plan");
    return VF_CUDA_OK;
}

int reserve_dag_plan_buffers(PersistentContext* context, const NativeDagPlan& plan) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    try {
        if (context->dag_u8.size() < plan.operators.size()) {
            context->dag_u8.resize(plan.operators.size(), nullptr);
            context->dag_u8_capacity.resize(plan.operators.size(), 0);
        }
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
    const size_t pixels = static_cast<size_t>(plan.width) * plan.height;
    bool needs_gaussian = false;
    bool needs_morph_scratch = false;
    size_t maximum_padded_count = 0;
    for (size_t index = 0; index < plan.operators.size(); ++index) {
        int result = reserve_device(
            &context->dag_u8[index], &context->dag_u8_capacity[index],
            pixels * static_cast<size_t>(plan.node_channels[index]), &context->allocation_count);
        if (result != VF_CUDA_OK) return result;
        const VfPlanOperatorV1& op = plan.operators[index];
        needs_gaussian = needs_gaussian || op.kind == VF_PLAN_GAUSSIAN;
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            int radius = 0, padded_width = 0, padded_height = 0;
            size_t padded_count = 0;
            result = adaptive_layout(plan.width, plan.height, op.int_params[0], &radius,
                                     &padded_width, &padded_height, &padded_count);
            if (result != VF_CUDA_OK) return result;
            maximum_padded_count = std::max(maximum_padded_count, padded_count);
        }
    }
    int result = reserve_device(&context->u8[0], &context->u8_capacity[0],
                                pixels * static_cast<size_t>(plan.input_channels),
                                &context->allocation_count);
    if (result == VF_CUDA_OK && needs_morph_scratch) result = reserve_device(
        &context->u8[4], &context->u8_capacity[4], pixels * 3, &context->allocation_count);
    if (result == VF_CUDA_OK && needs_gaussian) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        pixels * 3, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u8[3], &context->u8_capacity[3], maximum_padded_count, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[0], &context->u64_capacity[0], maximum_padded_count, &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[1], &context->u64_capacity[1], maximum_padded_count, &context->allocation_count);
    return result;
}

int reserve_plan_buffers(PersistentContext* context, const NativePlan& plan) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    const size_t input_pixels = static_cast<size_t>(plan.width) * plan.height;
    size_t maximum_pixels = input_pixels;
    int maximum_channels = plan.input_channels;
    bool needs_gaussian = false;
    bool needs_morph_scratch = false;
    size_t maximum_padded_count = 0;
    int channels = plan.input_channels;
    int current_width = plan.width;
    int current_height = plan.height;
    for (const VfPlanOperatorV1& op : plan.operators) {
        if (op.kind == VF_PLAN_GRAY) channels = 1;
        if (op.kind == VF_PLAN_RESIZE_AREA) {
            current_width = op.int_params[0];
            current_height = op.int_params[1];
        }
        const size_t current_pixels = static_cast<size_t>(current_width) * current_height;
        maximum_pixels = std::max(maximum_pixels, current_pixels);
        maximum_channels = std::max(maximum_channels, channels);
        needs_gaussian = needs_gaussian || op.kind == VF_PLAN_GAUSSIAN;
        needs_morph_scratch = needs_morph_scratch || op.kind == VF_PLAN_MORPHOLOGY;
        if (op.kind == VF_PLAN_ADAPTIVE_MEAN) {
            int radius = 0, padded_width = 0, padded_height = 0;
            size_t padded_count = 0;
            int result = adaptive_layout(current_width, current_height, op.int_params[0], &radius,
                                         &padded_width, &padded_height, &padded_count);
            if (result != VF_CUDA_OK) return result;
            maximum_padded_count = std::max(maximum_padded_count, padded_count);
        }
    }
    const size_t image_bytes = maximum_pixels * static_cast<size_t>(maximum_channels);
    int result = reserve_device(&context->u8[0], &context->u8_capacity[0], image_bytes,
                                &context->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &context->u8[1], &context->u8_capacity[1], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK) result = reserve_device(
        &context->u8[2], &context->u8_capacity[2], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK && needs_morph_scratch) result = reserve_device(
        &context->u8[4], &context->u8_capacity[4], image_bytes, &context->allocation_count);
    if (result == VF_CUDA_OK && needs_gaussian) result = reserve_device(
        &context->gaussian_buffer, &context->gaussian_capacity,
        maximum_pixels * static_cast<size_t>(maximum_channels), &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u8[3], &context->u8_capacity[3], maximum_padded_count,
        &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[0], &context->u64_capacity[0], maximum_padded_count,
        &context->allocation_count);
    if (result == VF_CUDA_OK && maximum_padded_count > 0) result = reserve_device(
        &context->u64[1], &context->u64_capacity[1], maximum_padded_count,
        &context->allocation_count);
    return result;
}

int cuda_result(cudaError_t error) { return visionflow_cuda::runtime_error(error); }

int alloc_copy(const uint8_t* host, int width, int height, int stride, int channels, uint8_t** device) {
    return visionflow_cuda::allocate_and_upload(host, width, height, stride, channels, device);
}

int copy_back_free(uint8_t* host, int stride, int width, int height, int channels, uint8_t* device) {
    return visionflow_cuda::download_and_free(host, stride, width, height, channels, device);
}

__device__ int reflect101(int value, int length) {
    if (length <= 1) return 0;
    while (value < 0 || value >= length) {
        value = value < 0 ? -value : 2 * length - value - 2;
    }
    return value;
}

__global__ void bgr_gray_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = (y * width + x) * 3;
    constexpr int gray_shift = 15;
    constexpr int blue_to_gray = 3735;
    constexpr int green_to_gray = 19235;
    constexpr int red_to_gray = 9798;
    dst[y * width + x] = static_cast<uint8_t>(
        (blue_to_gray * src[index] +
         green_to_gray * src[index + 1] +
         red_to_gray * src[index + 2] +
         (1 << (gray_shift - 1))) >> gray_shift);
}

// Grays a rectangular region of a wider BGR source into a tightly packed single-channel buffer.
// The plain bgr_gray_kernel above assumes a packed source, which a resident-image ROI is not.
__global__ void bgr_gray_roi_kernel(
    const uint8_t* src, int src_width, int offset_x, int offset_y,
    uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int index = ((y + offset_y) * src_width + x + offset_x) * 3;
    constexpr int gray_shift = 15;
    constexpr int blue_to_gray = 3735;
    constexpr int green_to_gray = 19235;
    constexpr int red_to_gray = 9798;
    dst[y * width + x] = static_cast<uint8_t>(
        (blue_to_gray * src[index] +
         green_to_gray * src[index + 1] +
         red_to_gray * src[index + 2] +
         (1 << (gray_shift - 1))) >> gray_shift);
}

__global__ void bgr_rgb_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = (y * width + x) * 3;
    dst[i] = src[i + 2]; dst[i + 1] = src[i + 1]; dst[i + 2] = src[i];
}

__global__ void crop_kernel(const uint8_t* src, uint8_t* dst, int src_width, int x0, int y0, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) dst[(y * width + x) * channels + c] = src[((y + y0) * src_width + x + x0) * channels + c];
}

// Exact OpenCV INTER_AREA downscale for CV_8UC1. Float accumulation order matches
// ResizeArea_Invoker; the DLL is built with --fmad=false so products are never fused.
__global__ void resize_area_kernel(
    const uint8_t* src, uint8_t* dst, int sw, int dw, int dh, int mode,
    int scale_x, int scale_y, float inverse_area, int x_entries,
    const int* indices, const float* alphas) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int value = 0;
    if (mode == AREA_RESIZE_COPY) {
        value = src[static_cast<size_t>(y) * sw + x];
    } else if (mode == AREA_RESIZE_FAST_2X2) {
        const uint8_t* row0 = src + static_cast<size_t>(y) * 2 * sw;
        const uint8_t* row1 = row0 + sw;
        const int sx = x * 2;
        value = (row0[sx] + row0[sx + 1] + row1[sx] + row1[sx + 1] + 2) >> 2;
    } else if (mode == AREA_RESIZE_FAST_INTEGER) {
        uint32_t sum = 0;
        for (int sy = 0; sy < scale_y; ++sy) {
            const uint8_t* row = src + (static_cast<size_t>(y) * scale_y + sy) * sw;
            for (int sx = 0; sx < scale_x; ++sx) sum += row[x * scale_x + sx];
        }
        float scaled = static_cast<float>(static_cast<int>(sum)) * inverse_area;
        value = static_cast<int>(nearbyintf(scaled));
    } else {
        const int y_offsets = dw + 1;
        const int x_sources = dw + dh + 2;
        const int y_sources = x_sources + x_entries;
        float total = 0.0f;
        for (int j = indices[y_offsets + y]; j < indices[y_offsets + y + 1]; ++j) {
            const uint8_t* row = src + static_cast<size_t>(indices[y_sources + j]) * sw;
            float row_sum = 0.0f;
            for (int k = indices[x]; k < indices[x + 1]; ++k) {
                float product = static_cast<float>(row[indices[x_sources + k]]) * alphas[k];
                row_sum = row_sum + product;
            }
            float weighted = alphas[x_entries + j] * row_sum;
            total = total + weighted;
        }
        value = static_cast<int>(nearbyintf(total));
    }
    dst[static_cast<size_t>(y) * dw + x] = static_cast<uint8_t>(max(0, min(255, value)));
}

__global__ void resize_gray_kernel(const uint8_t* src, uint8_t* dst, int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    // Upscale only; non-expanding targets use resize_area_kernel.
    float sx = (x + 0.5f) * sw / dw - 0.5f, sy = (y + 0.5f) * sh / dh - 0.5f;
    int raw_x0 = static_cast<int>(floorf(sx));
    int raw_y0 = static_cast<int>(floorf(sy));
    int x0 = max(0, min(sw - 1, raw_x0));
    int y0 = max(0, min(sh - 1, raw_y0));
    int x1 = max(0, min(sw - 1, raw_x0 + 1));
    int y1 = max(0, min(sh - 1, raw_y0 + 1));
    float ax = sx - floorf(sx), ay = sy - floorf(sy);
    float value = (1 - ay) * ((1 - ax) * src[y0 * sw + x0] + ax * src[y0 * sw + x1]) + ay * ((1 - ax) * src[y1 * sw + x0] + ax * src[y1 * sw + x1]);
    dst[y * dw + x] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void gaussian_horizontal_kernel(
    const uint8_t* src,
    uint32_t* intermediate,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uint8_t* row = src + static_cast<size_t>(y) * width * channels;
    const bool interior = x >= radius && x + radius < width;
    for (int c = 0; c < channels; ++c) {
        uint32_t sum = 0;
        if (interior) {
            const uint8_t* window = row + (x - radius) * channels + c;
            for (int k = 0; k <= 2 * radius; ++k) {
                sum += static_cast<uint32_t>(window[k * channels]) * gaussian_weights[k];
            }
        } else {
            for (int kx = -radius; kx <= radius; ++kx) {
                int sx = reflect101(x + kx, width);
                sum += static_cast<uint32_t>(row[sx * channels + c]) * gaussian_weights[kx + radius];
            }
        }
        intermediate[(static_cast<size_t>(y) * width + x) * channels + c] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const uint32_t* intermediate,
    uint8_t* dst,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t row_step = static_cast<size_t>(width) * channels;
    const bool interior = y >= radius && y + radius < height;
    for (int c = 0; c < channels; ++c) {
        unsigned long long sum = 0;
        if (interior) {
            const uint32_t* window = intermediate + (y - radius) * row_step + x * channels + c;
            for (int k = 0; k <= 2 * radius; ++k) {
                sum += static_cast<unsigned long long>(window[k * row_step]) * gaussian_weights[k];
            }
        } else {
            for (int ky = -radius; ky <= radius; ++ky) {
                int sy = reflect101(y + ky, height);
                sum += static_cast<unsigned long long>(
                    intermediate[sy * row_step + x * channels + c]) * gaussian_weights[ky + radius];
            }
        }
        dst[(static_cast<size_t>(y) * width + x) * channels + c] = static_cast<uint8_t>(
            (sum + GAUSSIAN_FINAL_ROUND) >>
            (GAUSSIAN_FIXED_SHIFT * 2));
    }
}

__global__ void threshold_kernel(const uint8_t* src, uint8_t* dst, int count, int threshold, int max_value, int invert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool high = src[i] > threshold;
    dst[i] = static_cast<uint8_t>((invert ? !high : high) ? max_value : 0);
}

__global__ void replicate_border_kernel(
    const uint8_t* src,
    uint8_t* padded,
    int width,
    int height,
    int padded_width,
    int padded_height,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padded_width || y >= padded_height) return;
    int source_x = max(0, min(width - 1, x - radius));
    int source_y = max(0, min(height - 1, y - radius));
    padded[y * padded_width + x] = src[source_y * width + source_x];
}

__global__ void row_prefix_u8_kernel(
    const uint8_t* src,
    unsigned long long* prefix,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? static_cast<unsigned long long>(src[row * width + column]) : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) prefix[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__global__ void transpose_u64_kernel(
    const unsigned long long* src,
    unsigned long long* dst,
    int width,
    int height) {
    __shared__ unsigned long long tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];
    int x = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < width && y + offset < height) {
            tile[threadIdx.y + offset][threadIdx.x] = src[(y + offset) * width + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < height && y + offset < width) {
            dst[(y + offset) * height + x] = tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void row_prefix_u64_inplace_kernel(
    unsigned long long* values,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? values[row * width + column] : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) values[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__device__ unsigned long long integral_value_transposed(
    const unsigned long long* integral_transposed,
    int padded_height,
    int x,
    int y) {
    if (x < 0 || y < 0) return 0ULL;
    return integral_transposed[x * padded_height + y];
}

__global__ void adaptive_integral_kernel(
    const uint8_t* src,
    const unsigned long long* integral_transposed,
    uint8_t* dst,
    int width,
    int height,
    int padded_height,
    int block_size,
    float c,
    int max_value,
    int invert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int x0 = x;
    int y0 = y;
    int x1 = x + block_size - 1;
    int y1 = y + block_size - 1;
    unsigned long long bottom_right =
        integral_value_transposed(integral_transposed, padded_height, x1, y1);
    unsigned long long above =
        integral_value_transposed(integral_transposed, padded_height, x1, y0 - 1);
    unsigned long long left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y1);
    unsigned long long above_left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y0 - 1);
    unsigned long long sum = (bottom_right + above_left) - (above + left);
    unsigned long long area = static_cast<unsigned long long>(block_size) * block_size;
    int mean = static_cast<int>((sum + area / 2ULL) / area);
    bool selected = invert
        ? static_cast<int>(src[y * width + x]) <= mean - static_cast<int>(floorf(c))
        : static_cast<int>(src[y * width + x]) > mean - static_cast<int>(ceilf(c));
    dst[y * width + x] = static_cast<uint8_t>(selected ? max_value : 0);
}

__global__ void morph_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int radius, int dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        int value = dilate ? 0 : 255;
        for (int ky = -radius; ky <= radius; ++ky) for (int kx = -radius; kx <= radius; ++kx) {
            int sx = x + kx, sy = y + ky;
            int sample = (sx < 0 || sx >= width || sy < 0 || sy >= height) ? (dilate ? 0 : 255) : src[(sy * width + sx) * channels + c];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        dst[(y * width + x) * channels + c] = static_cast<uint8_t>(value);
    }
}

// A rectangular min/max filter is separable. Keep both directions in one
// block so a 5x5 pass needs only one launch and no device scratch image.
__global__ void morph_k5_shared_kernel(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels, int dilate) {
    constexpr int radius = 2;
    constexpr int tile_width = BLOCK_X + 2 * radius;
    constexpr int tile_height = BLOCK_Y + 2 * radius;
    __shared__ uint8_t tile[tile_width * tile_height * 3];
    __shared__ uint8_t horizontal[tile_height * BLOCK_X * 3];
    const int thread_index = threadIdx.y * BLOCK_X + threadIdx.x;
    const int thread_count = BLOCK_X * BLOCK_Y;
    const uint8_t border = static_cast<uint8_t>(dilate ? 0 : 255);

    for (int index = thread_index; index < tile_width * tile_height * channels;
         index += thread_count) {
        const int channel = index % channels;
        const int pixel = index / channels;
        const int x = blockIdx.x * BLOCK_X + pixel % tile_width - radius;
        const int y = blockIdx.y * BLOCK_Y + pixel / tile_width - radius;
        tile[index] = (x < 0 || x >= width || y < 0 || y >= height)
            ? border : src[(y * width + x) * channels + channel];
    }
    __syncthreads();

    for (int index = thread_index; index < tile_height * BLOCK_X * channels;
         index += thread_count) {
        const int channel = index % channels;
        const int pixel = index / channels;
        const int x = pixel % BLOCK_X + radius;
        const int y = pixel / BLOCK_X;
        int value = dilate ? 0 : 255;
        for (int dx = -radius; dx <= radius; ++dx) {
            const int sample = tile[(y * tile_width + x + dx) * channels + channel];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        horizontal[index] = static_cast<uint8_t>(value);
    }
    __syncthreads();

    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int channel = 0; channel < channels; ++channel) {
        int value = dilate ? 0 : 255;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int sample = horizontal[
                ((threadIdx.y + dy + radius) * BLOCK_X + threadIdx.x) * channels + channel];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        dst[(y * width + x) * channels + channel] = static_cast<uint8_t>(value);
    }
}

dim3 grid2d(int width, int height);

void launch_morph_pass(
    const uint8_t* src, uint8_t* dst, int width, int height, int channels,
    int radius, int dilate, cudaStream_t stream = nullptr) {
    if (radius == 2) {
        morph_k5_shared_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
            src, dst, width, height, channels, dilate);
    } else {
        morph_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
            src, dst, width, height, channels, radius, dilate);
    }
}

// Separable Gaussian. A block-local shared-memory tile variant produced identical output but
// was about 2x slower on RTX 3090 (2026-09-14), so both passes read global memory directly.
void launch_gaussian(
    const uint8_t* src, uint32_t* intermediate, uint8_t* dst, int width, int height,
    int channels, int radius, cudaStream_t stream = nullptr) {
    gaussian_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, intermediate, width, height, channels, radius);
    gaussian_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        intermediate, dst, width, height, channels, radius);
}

// float32 separable Gaussian for the optional vf_gaussian_blur_f32 export. Single channel only:
// the caller passes the detector's float32 gray/residual plane. Taps are read from the constant
// coefficient table prepared by prepare_gaussian_f32_weights(), and both passes accumulate in
// float32 with reflect101 borders, which is what cv2.GaussianBlur(single_channel_float32, ksize,
// 0.0) does internally. The build disables FMA contraction (/fmad=false), so the result is a pure
// function of the tap order documented here and of the input bytes.
//
// Horizontal pass: taps are accumulated left to right, the order OpenCV's row filter uses.
__global__ void gaussian_f32_horizontal_kernel(
    const float* src, float* dst, int width, int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const float* row = src + static_cast<size_t>(y) * width;
    float sum = 0.0f;
    if (x >= radius && x + radius < width) {
        const float* window = row + (x - radius);
        for (int k = 0; k <= 2 * radius; ++k) {
            sum += window[k] * gaussian_f32_weights[k];
        }
    } else {
        for (int kx = -radius; kx <= radius; ++kx) {
            sum += row[reflect101(x + kx, width)] * gaussian_f32_weights[kx + radius];
        }
    }
    dst[static_cast<size_t>(y) * width + x] = sum;
}

// Vertical pass: symmetric taps are added as a pair before the multiply, the order OpenCV's
// symmetric column filter uses. Measured against cv2.GaussianBlur this is closer than a plain
// left-to-right sum for every kernel size in the verified range (see the equivalence tool).
__global__ void gaussian_f32_vertical_kernel(
    const float* src, float* dst, int width, int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::ptrdiff_t step = static_cast<std::ptrdiff_t>(width);
    float sum = 0.0f;
    if (y >= radius && y + radius < height) {
        const float* centre = src + static_cast<size_t>(y) * width + x;
        sum = centre[0] * gaussian_f32_weights[radius];
        for (int k = 1; k <= radius; ++k) {
            const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(k) * step;
            sum += (centre[-offset] + centre[offset]) * gaussian_f32_weights[radius + k];
        }
    } else {
        sum = src[static_cast<size_t>(y) * width + x] * gaussian_f32_weights[radius];
        for (int k = 1; k <= radius; ++k) {
            const float top = src[static_cast<size_t>(reflect101(y - k, height)) * width + x];
            const float bottom = src[static_cast<size_t>(reflect101(y + k, height)) * width + x];
            sum += (top + bottom) * gaussian_f32_weights[radius + k];
        }
    }
    dst[static_cast<size_t>(y) * width + x] = sum;
}

void launch_gaussian_f32(
    const float* src, float* intermediate, float* dst, int width, int height,
    int radius, cudaStream_t stream) {
    gaussian_f32_horizontal_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, intermediate, width, height, radius);
    gaussian_f32_vertical_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        intermediate, dst, width, height, radius);
}

__global__ void gather_roi_batch_kernel(
    const uint8_t* source,
    int source_width,
    int channels,
    const VfRoiV1* rois,
    uint8_t* batch,
    int roi_width,
    int roi_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int roi_index = blockIdx.z;
    if (x >= roi_width || y >= roi_height) return;
    const VfRoiV1 roi = rois[roi_index];
    size_t source_pixel =
        (static_cast<size_t>(roi.y + y) * source_width + roi.x + x) * channels;
    size_t batch_pixel =
        ((static_cast<size_t>(roi_index) * roi_height + y) * roi_width + x) * channels;
    for (int channel = 0; channel < channels; ++channel) {
        batch[batch_pixel + channel] = source[source_pixel + channel];
    }
}

dim3 grid2d(int width, int height) { return dim3((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y); }

// --- Template Anchor Grid localization (TM_CCOEFF_NORMED) -------------------------------
// Reproduces core/tiler.py::Tiler._find_grid_anchor: the same correlation formula and the same
// "topmost then leftmost wins a tie" order as the CPU argmax over the result map. Only the
// match rectangle and its score cross PCIe.
//
// Two vertical prefixes per output column give the window pixel sum and the window square sum in
// O(1) each. The template-weighted window sum is accumulated directly by the score kernel:
//   num = sum(w*t) - N*mean_w*mean_t
//   den = sqrt((sum(w^2) - N*mean_w^2) * (sum(t^2) - N*mean_t^2))
constexpr int MATCH_TEMPLATE_BUFFER = 2;    // context->u8[2]: template gray on device
constexpr int MATCH_ROI_BUFFER = 3;         // context->u8[3]: search ROI gray on device
constexpr int MATCH_REDUCE_BLOCK = 256;
// Coordinates are packed into 20 bits, so a search wider than this cannot be reported at all.
constexpr int MATCH_MAX_OUTPUT_WIDTH = (1 << 20) - 1;
// Output tile of the tiled score kernel and its block shape. Shared memory per block is
// (tile_rows + template_height - 1) * (tile_cols + template_width - 1) bytes, so the caller
// shrinks the tile when a large template would exceed the device limit.
constexpr int MATCH_TILE_COLS = 64;
constexpr int MATCH_TILE_ROWS = 32;
constexpr int MATCH_BLOCK_X = 64;
constexpr int MATCH_BLOCK_Y = 4;
// sm_86 allows up to 100 KiB of dynamic shared memory per block once opted in, and the export
// opts in for every launch it makes. The tile shrink loop stays within this budget.
constexpr size_t MATCH_SHARED_LIMIT_BYTES = 99 * 1024;
// Candidate slots scale with the search width, not the block size: the score kernel writes one
// entry per output column, so a fixed small count would overflow as soon as output_width exceeds
// it. Slots are int64-sized so the double scores stay aligned, and the result fields follow them.
// Layout, all indexed by column count:
//   [0, W)      packed best keys (one per output column, int64, atomic target)
//   [W, 2W)     candidate score (double, one per output column)
//   [2W, 3W)    candidate row (int, one per output column, padded to int64 stride)
//   then        the two coordinate ints, the float score and the global best key
constexpr int MATCH_CANDIDATE_SLOT_STRIDE = 3;
constexpr int MATCH_RESULT_SLOTS = 3;
constexpr int MATCH_FIXED_SLOTS = MATCH_RESULT_SLOTS + 1;

int match_candidate_slots(int output_width) {
    return output_width > 0 ? output_width : 1;
}

int match_slot_count(int output_width) {
    return match_candidate_slots(output_width) * MATCH_CANDIDATE_SLOT_STRIDE + MATCH_FIXED_SLOTS;
}

int match_score_offset(int output_width) {
    return match_candidate_slots(output_width);
}

int match_row_offset(int output_width) {
    return match_candidate_slots(output_width) * 2;
}

int match_result_offset(int output_width) {
    return match_candidate_slots(output_width) * MATCH_CANDIDATE_SLOT_STRIDE;
}
// Match key layout, ordered so that a larger unsigned key is the better match:
//   bits 62..40 score, bits 39..20 inverted y, bits 19..0 inverted x.
// The score is quantized to 22 bits once, and that same value is what the caller receives, so a
// tie in the packed key is exactly a tie in the reported score. Coordinates must fit 20 bits.
constexpr int MATCH_SCORE_BITS = 22;
constexpr double MATCH_SCORE_MAX = static_cast<double>((1 << MATCH_SCORE_BITS) - 1);
constexpr int MATCH_COORD_BITS = 20;
constexpr int MATCH_COORD_MASK = (1 << MATCH_COORD_BITS) - 1;

__device__ __forceinline__ unsigned long long match_pack_key(double score, int x, int y) {
    double clamped = score;
    if (clamped > 1.0) clamped = 1.0;
    if (clamped < 0.0) clamped = 0.0;
    const unsigned long long quantized =
        static_cast<unsigned long long>(clamped * MATCH_SCORE_MAX + 0.5);
    return (quantized << 40) |
           (static_cast<unsigned long long>(MATCH_COORD_MASK - y) << 20) |
           static_cast<unsigned long long>(MATCH_COORD_MASK - x);
}

__device__ __forceinline__ void match_unpack_key(
    unsigned long long key, double* score, int* x, int* y) {
    *x = MATCH_COORD_MASK - static_cast<int>(key & MATCH_COORD_MASK);
    *y = MATCH_COORD_MASK - static_cast<int>((key >> 20) & MATCH_COORD_MASK);
    const unsigned long long quantized = (key >> 40) & ((1ULL << MATCH_SCORE_BITS) - 1);
    *score = static_cast<double>(quantized) / MATCH_SCORE_MAX;
}

// One thread per output column. Writes the vertical prefixes of the horizontal window sum and of
// its square, so the score kernel obtains the window pixel sum and the window square sum with two
// subtractions each. The horizontal window for output (row, column) starts at image row `row`.
__global__ void match_prefix_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int template_width,
    long long* sum_prefix, long long* square_prefix) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    long long sum_running = 0;
    long long square_running = 0;
    for (int row = threadIdx.y; row < roi_height; row += blockDim.y) {
        const uint8_t* line = roi + static_cast<size_t>(row) * roi_width;
        int sum = 0;
        long long square_sum = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const int value = line[column + offset];
            sum += value;
            square_sum += static_cast<long long>(value) * value;
        }
        sum_running += sum;
        square_running += square_sum;
        const size_t index = static_cast<size_t>(row) * output_width + column;
        sum_prefix[index] = sum_running;
        square_prefix[index] = square_running;
    }
}

// Offers one candidate for a column through a dedicated packed-key slot. The key is
// (quantized score, inverted row, inverted column), so a larger key is a better match and a tie
// resolves to the smaller row and then the smaller column - the same order as the CPU argmax over
// the match map. Publishing the whole triple with one compare-and-swap on its own array keeps the
// winner independent of block and thread order; the score array is never used as the atomic.
__device__ __forceinline__ void match_offer_candidate(
    double score, int column, int row, unsigned long long* best_keys) {
    double clamped = score;
    if (clamped > 1.0) clamped = 1.0;
    if (clamped < 0.0) clamped = 0.0;
    const unsigned long long quantized =
        static_cast<unsigned long long>(clamped * MATCH_SCORE_MAX + 0.5);
    const int safe_row = row > MATCH_COORD_MASK - 1 ? MATCH_COORD_MASK - 1 : row;
    const int safe_column = column > MATCH_COORD_MASK - 1 ? MATCH_COORD_MASK - 1 : column;
    const unsigned long long key =
        (quantized << 40) |
        (static_cast<unsigned long long>(MATCH_COORD_MASK - safe_row) << 20) |
        static_cast<unsigned long long>(MATCH_COORD_MASK - safe_column);
    unsigned long long* slot = best_keys + column;
    unsigned long long current = *slot;
    while (key > current) {
        const unsigned long long previous = atomicCAS(slot, current, key);
        if (previous == current) break;
        current = previous;
    }
}

// One thread per output row within a column band. The template is staged in shared memory, because
// it is the small, constant, every-candidate operand; the ROI stays in global memory and each row
// the thread reads is reused across all template rows it participates in, so the cost per output
// is one read of a template_width strip plus one multiply-add chain. Consecutive threads handle
// consecutive output rows, so both the ROI loads and the template loads coalesce (the warp shares
// one template row and reads a contiguous ROI patch).
__global__ void match_score_shared_template_kernel(
    const uint8_t* roi, int roi_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    int column_band,
    double template_mean, double template_variance,
    const uint8_t* templ, unsigned long long* best_keys) {
    extern __shared__ unsigned char shared_bytes[];
    uint8_t* shared_template = shared_bytes;

    for (int index = threadIdx.y * blockDim.x + threadIdx.x;
         index < template_width * template_height; index += blockDim.x * blockDim.y) {
        shared_template[index] = templ[index];
    }
    __syncthreads();

    const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * column_band + threadIdx.x;
    if (column >= output_width || output_row >= output_height) return;

    long long window_sum = 0;
    long long window_square = 0;
    long long weighted = 0;
    for (int template_row = 0; template_row < template_height; ++template_row) {
        const uint8_t* image_line =
            roi + static_cast<size_t>(output_row + template_row) * roi_width + column;
        const uint8_t* template_line =
            shared_template + static_cast<size_t>(template_row) * template_width;
        long long term = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const int value = image_line[offset];
            window_sum += value;
            window_square += static_cast<long long>(value) * value;
            term += static_cast<long long>(value) * template_line[offset];
        }
        weighted += term;
    }
    const double mean = static_cast<double>(window_sum) / template_pixels;
    double window_variance = static_cast<double>(window_square) / template_pixels - mean * mean;
    if (window_variance < 0.0) window_variance = 0.0;
    const double denominator = std::sqrt(window_variance * template_variance) * template_pixels;
    double score = -1.0;
    if (denominator > 0.0) {
        score = (static_cast<double>(weighted) - template_mean * window_sum) / denominator;
    }
    if (score > 1.0) score = 1.0;
    if (score < -1.0) score = -1.0;
    // Several output rows of the same column are in flight at once, so publish through the same
    // packed-key compare-and-swap the tiled kernel uses; the ordering is identical, so the winner
    // does not depend on which block ran first.
    match_offer_candidate(score, column, output_row, best_keys);
}

// Tiled score kernel. Each block stages the ROI patch covering its output tile into shared memory
// once, so the window sum, its square and the template-weighted sum are all accumulated from
// shared memory instead of re-reading the ROI for every candidate. The template stays in global
// memory on purpose: it is small, constant per call, and reused by every block, so it stays hot in
// L2 without competing with the ROI patch for shared memory.
//
// Shared bytes: (tile_rows + template_height - 1) * (tile_cols + template_width - 1).
__global__ void match_score_kernel(
    const uint8_t* roi, int roi_width,
    int output_width, int output_height,
    int template_width, int template_height, int template_pixels,
    int tile_cols, int tile_rows,
    double template_mean, double template_variance,
    const uint8_t* templ,
    unsigned long long* best_keys) {
    extern __shared__ unsigned char shared_bytes[];
    uint8_t* tile = shared_bytes;

    const int tiles_x = (output_width + tile_cols - 1) / tile_cols;
    const int tile_x = blockIdx.x % tiles_x;
    const int tile_y = blockIdx.x / tiles_x;
    const int first_column = tile_x * tile_cols;
    const int first_row = tile_y * tile_rows;
    const int columns = min(tile_cols, output_width - first_column);
    const int rows = min(tile_rows, output_height - first_row);
    const int patch_rows = rows + template_height - 1;
    const int patch_cols = columns + template_width - 1;
    const int pitch = patch_cols;

    for (int row = threadIdx.y; row < patch_rows; row += blockDim.y) {
        const uint8_t* source =
            roi + static_cast<size_t>(first_row + row) * roi_width + first_column;
        uint8_t* destination = tile + static_cast<size_t>(row) * pitch;
        for (int column = threadIdx.x; column < patch_cols; column += blockDim.x) {
            destination[column] = source[column];
        }
    }
    __syncthreads();

    for (int output_row = threadIdx.y; output_row < rows; output_row += blockDim.y) {
        for (int output_column = threadIdx.x; output_column < columns; output_column += blockDim.x) {
            long long window_sum = 0;
            long long window_square = 0;
            long long weighted = 0;
            for (int template_row = 0; template_row < template_height; ++template_row) {
                const uint8_t* patch =
                    tile + static_cast<size_t>(output_row + template_row) * pitch + output_column;
                const uint8_t* template_line =
                    templ + static_cast<size_t>(template_row) * template_width;
                for (int offset = 0; offset < template_width; ++offset) {
                    const int value = patch[offset];
                    window_sum += value;
                    window_square += static_cast<long long>(value) * value;
                    weighted += static_cast<long long>(value) * template_line[offset];
                }
            }
            const double mean = static_cast<double>(window_sum) / template_pixels;
            double window_variance =
                static_cast<double>(window_square) / template_pixels - mean * mean;
            if (window_variance < 0.0) window_variance = 0.0;
            const double denominator =
                std::sqrt(window_variance * template_variance) * template_pixels;
            double score = -1.0;
            if (denominator > 0.0) {
                score = (static_cast<double>(weighted) - template_mean * window_sum) / denominator;
            }
            if (score > 1.0) score = 1.0;
            if (score < -1.0) score = -1.0;
            match_offer_candidate(
                score, first_column + output_column, first_row + output_row, best_keys);
        }
    }
}


// One thread per output column. Builds the vertical prefix of the ROI pixel sums and of the ROI
// sum of squares, which is everything the TM_SQDIFF_NORMED score needs:
//   sum((w - t)^2) = sum(w^2) - 2*sum(w*t) + sum(t^2)
__global__ void match_diff_prefix_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int template_width,
    long long* value_prefix, long long* square_prefix) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    long long value_running = 0;
    long long square_running = 0;
    for (int row = threadIdx.y; row < roi_height; row += blockDim.y) {
        const uint8_t* line = roi + static_cast<size_t>(row) * roi_width;
        long long value_sum = 0;
        long long square_sum = 0;
        for (int offset = 0; offset < template_width; ++offset) {
            const long long value = line[column + offset];
            value_sum += value;
            square_sum += value * value;
        }
        value_running += value_sum;
        square_running += square_sum;
        const size_t index = static_cast<size_t>(row) * output_width + column;
        value_prefix[index] = value_running;
        square_prefix[index] = square_running;
    }
}

// One thread per output column minimises the normalized squared difference over the column. The
// packed value is 1 - difference so that the shared "larger key wins" reduction, and therefore the
// topmost-then-leftmost tie order, describes the smallest difference.
__global__ void match_diff_score_kernel(
    const uint8_t* roi, int roi_width, int roi_height,
    int output_width, int output_height,
    int template_width, int template_height,
    const uint8_t* templ,
    const long long* value_prefix, const long long* square_prefix,
    long long template_square_sum,
    double* candidate_scores, int* candidate_ys) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width || template_height > roi_height) return;
    double best_value = -1.0;
    int best_y = -1;
    for (int row = 0; row < output_height; ++row) {
        const size_t bottom = static_cast<size_t>(row + template_height - 1) * output_width + column;
        long long window_value = value_prefix[bottom];
        long long window_square = square_prefix[bottom];
        if (row > 0) {
            const size_t top = static_cast<size_t>(row - 1) * output_width + column;
            window_value -= value_prefix[top];
            window_square -= square_prefix[top];
        }
        long long cross = 0;
        for (int template_row = 0; template_row < template_height; ++template_row) {
            const uint8_t* line =
                roi + static_cast<size_t>(row + template_row) * roi_width + column;
            const uint8_t* template_line =
                templ + static_cast<size_t>(template_row) * template_width;
            long long term = 0;
            for (int offset = 0; offset < template_width; ++offset) {
                term += static_cast<long long>(line[offset]) * template_line[offset];
            }
            cross += term;
        }
        const double numerator =
            static_cast<double>(window_square) - 2.0 * cross + static_cast<double>(template_square_sum);
        const double denominator = std::sqrt(
            static_cast<double>(window_square) * static_cast<double>(template_square_sum));
        const double difference = denominator > 0.0 ? numerator / denominator : 1.0;
        const double clamped_difference = difference < 0.0 ? 0.0 : (difference > 1.0 ? 1.0 : difference);
        const double value = 1.0 - clamped_difference;
        if (value > best_value) {
            best_value = value;
            best_y = row;
        }
    }
    candidate_scores[column] = best_value;
    candidate_ys[column] = best_y;
}

// Reduces the per-column packed keys into one slot, keeping the same ordering the columns used.
__global__ void match_publish_kernel(
    const unsigned long long* column_keys, int output_width, unsigned long long* best_key) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= output_width) return;
    const unsigned long long key = column_keys[column];
    if (key == 0) return;
    unsigned long long* slot = best_key;
    unsigned long long current = *slot;
    while (key > current) {
        const unsigned long long previous = atomicCAS(slot, current, key);
        if (previous == current) break;
        current = previous;
    }
}

// Expands the winning key back into the rectangle origin and score. mode_sign is +1 for
// TM_CCOEFF_NORMED (report the packed value) and -1 for TM_SQDIFF_NORMED (report 1 - value), the
// same conversion core/tiler.py applies when it detects a flat template.
__global__ void match_unpack_kernel(
    const unsigned long long* best_key, float mode_sign, int* out_xy, float* out_score) {
    double score = 0.0;
    int x = 0;
    int y = 0;
    match_unpack_key(*best_key, &score, &x, &y);
    out_xy[0] = x;
    out_xy[1] = y;
    *out_score = static_cast<float>(1.0 + mode_sign * (score - 1.0));
}


// Mirrors OpenCV computeResizeAreaTab: double geometry, 1e-3 edge tolerance, float weights.
void append_area_axis(
    int source_size, int target_size, double scale,
    std::vector<int>* offsets, std::vector<int>* sources, std::vector<float>* weights) {
    offsets->push_back(0);
    for (int target = 0; target < target_size; ++target) {
        const double first = target * scale;
        const double last = first + scale;
        const double cell_width = std::min(scale, source_size - first);
        int start = static_cast<int>(std::ceil(first));
        int end = static_cast<int>(std::floor(last));
        end = std::min(end, source_size - 1);
        start = std::min(start, end);
        if (start - first > 1e-3) {
            sources->push_back(start - 1);
            weights->push_back(static_cast<float>((start - first) / cell_width));
        }
        for (int source = start; source < end; ++source) {
            sources->push_back(source);
            weights->push_back(static_cast<float>(1.0 / cell_width));
        }
        if (last - end > 1e-3) {
            sources->push_back(end);
            weights->push_back(static_cast<float>(
                std::min(std::min(last - end, 1.0), cell_width) / cell_width));
        }
        offsets->push_back(static_cast<int>(sources->size()));
    }
}

int prepare_area_resize(
    int source_width, int source_height, int target_width, int target_height,
    AreaResizeTables* tables, unsigned long long* allocation_count) {
    if (tables == nullptr || source_width <= 0 || source_height <= 0 || target_width <= 0 ||
        target_height <= 0 || target_width > source_width || target_height > source_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (source_width == target_width && source_height == target_height) {
        tables->mode = AREA_RESIZE_COPY;
        return VF_CUDA_OK;
    }
    const double scale_x = 1.0 / (static_cast<double>(target_width) / source_width);
    const double scale_y = 1.0 / (static_cast<double>(target_height) / source_height);
    const int integer_x = static_cast<int>(std::lround(scale_x));
    const int integer_y = static_cast<int>(std::lround(scale_y));
    if (std::abs(scale_x - integer_x) < DBL_EPSILON && std::abs(scale_y - integer_y) < DBL_EPSILON) {
        tables->scale_x = integer_x;
        tables->scale_y = integer_y;
        if (integer_x == 2 && integer_y == 2) {
            tables->mode = AREA_RESIZE_FAST_2X2;
        } else {
            tables->mode = AREA_RESIZE_FAST_INTEGER;
            tables->inverse_area = static_cast<float>(1.0 / (integer_x * integer_y));
        }
        return VF_CUDA_OK;
    }
    std::vector<int> x_offsets, y_offsets, x_sources, y_sources;
    std::vector<float> x_weights, y_weights;
    try {
        append_area_axis(source_width, target_width, scale_x, &x_offsets, &x_sources, &x_weights);
        append_area_axis(source_height, target_height, scale_y, &y_offsets, &y_sources, &y_weights);
        std::vector<int> indices;
        indices.reserve(x_offsets.size() + y_offsets.size() + x_sources.size() + y_sources.size());
        indices.insert(indices.end(), x_offsets.begin(), x_offsets.end());
        indices.insert(indices.end(), y_offsets.begin(), y_offsets.end());
        indices.insert(indices.end(), x_sources.begin(), x_sources.end());
        indices.insert(indices.end(), y_sources.begin(), y_sources.end());
        std::vector<float> weights(x_weights);
        weights.insert(weights.end(), y_weights.begin(), y_weights.end());

        tables->mode = AREA_RESIZE_GENERAL;
        tables->x_entries = static_cast<int>(x_sources.size());
        int* device_indices = nullptr;
        cudaError_t error = cudaMalloc(&device_indices, indices.size() * sizeof(int));
        if (error != cudaSuccess) return cuda_result(error);
        tables->indices = device_indices;
        if (allocation_count != nullptr) ++(*allocation_count);
        float* device_alphas = nullptr;
        error = cudaMalloc(&device_alphas, weights.size() * sizeof(float));
        if (error != cudaSuccess) return cuda_result(error);
        tables->alphas = device_alphas;
        if (allocation_count != nullptr) ++(*allocation_count);
        error = cudaMemcpy(tables->indices, indices.data(), indices.size() * sizeof(int),
                           cudaMemcpyHostToDevice);
        if (error == cudaSuccess) {
            error = cudaMemcpy(tables->alphas, weights.data(), weights.size() * sizeof(float),
                               cudaMemcpyHostToDevice);
        }
        return cuda_result(error);
    } catch (const std::bad_alloc&) {
        return VF_CUDA_ALLOCATION_FAILED;
    }
}

void launch_area_resize(
    const AreaResizeTables& tables, const uint8_t* src, uint8_t* dst,
    int source_width, int target_width, int target_height, cudaStream_t stream = nullptr) {
    resize_area_kernel<<<grid2d(target_width, target_height), dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
        src, dst, source_width, target_width, target_height, tables.mode, tables.scale_x,
        tables.scale_y, tables.inverse_area, tables.x_entries, tables.indices, tables.alphas);
}

// ---------------------------------------------------------------------------------------------
// Contour extension: cv2.findContours(RETR_LIST | RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) equivalence.
//
// This is a direct port of tools/contour_reference.py, which was verified point-for-point against
// cv2.findContours. The reference follows OpenCV contours.cpp:
//   cvStartFindContours_Impl -> 1-pixel zero frame, THRESH_BINARY binarization, scanner state
//   cvFindNextContour        -> raster scan, outer/hole classification, lnbd bookkeeping
//   icvFetchContour          -> the 8-neighbour border trace, CHAIN_APPROX_SIMPLE point rule
//
// Two properties keep the port cheap and exact:
//   - CHAIN_APPROX_SIMPLE compression is inherent to the trace (a point is emitted only where the
//     step direction changes), so no separate compression pass exists.
//   - The trace only ever tests whether a neighbour is non-zero, and marking only rewrites 1 into
//     2 or -126 (both still non-zero), so the trace of a border is independent of the marks left
//     by other borders. Only the raster scan is order-dependent, which is why it stays serial.
//
// OpenCV reports the flat contour list in reverse discovery order (icvEndProcessContour prepends to
// frame->v_next), so a final kernel reverses the discovery-order scratch into the output.
// ---------------------------------------------------------------------------------------------
constexpr int CONTOUR_NBD = 2;        // const schar nbd = 2 inside icvFetchContour
constexpr int CONTOUR_MARKED = -126;  // (schar)(nbd | -128)

// CV_INIT_3X3_DELTAS(deltas, step, 1): index 0..7 is E, NE, N, NW, W, SW, S, SE and 8..15 mirrors
// it. `index & 7` reproduces the 16-entry table exactly, including the exhausted search (15).
__device__ __forceinline__ void contour_ring_step(int index, int* dy, int* dx) {
    const int ring_dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int ring_dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    const int slot = index & 7;
    *dx = ring_dx[slot];
    *dy = ring_dy[slot];
}

// Appends one point; a full buffer sets the overflow flag instead of truncating the contour.
__device__ __forceinline__ void contour_store_point(
    int32_t* points, int point_capacity, int* point_index, int* overflow, int px, int py) {
    const int index = *point_index;
    if (index < point_capacity) {
        points[static_cast<size_t>(index) * 2] = px;
        points[static_cast<size_t>(index) * 2 + 1] = py;
    } else {
        *overflow = 1;
    }
    *point_index = index + 1;
}

// Port of icvFetchContour(ptr, step, pt, contour, CV_CHAIN_APPROX_SIMPLE). The padded label image
// is mutated in place exactly like OpenCV does (marks 2 / -126).
__device__ void contour_fetch(
    signed char* image, int stride, int i0_y, int i0_x, int is_hole, int pt_x, int pt_y,
    int32_t* points, int point_capacity, int* point_index, int* overflow) {
    int s_end = is_hole ? 0 : 4;
    int s = s_end;
    int i1_y = i0_y;
    int i1_x = i0_x;
    for (;;) {
        s = (s - 1) & 7;
        int dy = 0;
        int dx = 0;
        contour_ring_step(s, &dy, &dx);
        i1_y = i0_y + dy;
        i1_x = i0_x + dx;
        if (image[static_cast<size_t>(i1_y) * stride + i1_x] != 0) break;
        if (s == s_end) break;
    }

    if (s == s_end) {
        // Single-pixel domain: mark the pixel and emit exactly one point.
        image[static_cast<size_t>(i0_y) * stride + i0_x] = static_cast<signed char>(CONTOUR_MARKED);
        contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
        return;
    }

    int i3_y = i0_y;
    int i3_x = i0_x;
    int prev_s = s ^ 4;
    int i4_y = 0;
    int i4_x = 0;
    for (;;) {
        s_end = s;
        // `s` is always in 0..7 here, so C's `s = min(s, MAX_SIZE - 1)` is a no-op.
        while (s < 15) {
            s += 1;
            int dy = 0;
            int dx = 0;
            contour_ring_step(s, &dy, &dx);
            i4_y = i3_y + dy;
            i4_x = i3_x + dx;
            if (image[static_cast<size_t>(i4_y) * stride + i4_x] != 0) break;
        }
        s &= 7;

        // Right-bound marking: (unsigned)(s - 1) < (unsigned)s_end means 1 <= s <= s_end.
        if (s >= 1 && (s - 1) < s_end) {
            image[static_cast<size_t>(i3_y) * stride + i3_x] =
                static_cast<signed char>(CONTOUR_MARKED);
        } else if (image[static_cast<size_t>(i3_y) * stride + i3_x] == 1) {
            image[static_cast<size_t>(i3_y) * stride + i3_x] = static_cast<signed char>(CONTOUR_NBD);
        }

        if (s != prev_s) {
            contour_store_point(points, point_capacity, point_index, overflow, pt_x, pt_y);
            prev_s = s;
        }

        int dy = 0;
        int dx = 0;
        contour_ring_step(s, &dy, &dx);
        pt_y += dy;
        pt_x += dx;

        if (i4_y == i0_y && i4_x == i0_x && i3_y == i1_y && i3_x == i1_x) break;
        i3_y = i4_y;
        i3_x = i4_x;
        s = (s + 4) & 7;
    }
}

// Builds the 1-pixel-zero-framed label image of the requested region. Every padded pixel is
// written by exactly one thread, so the buffer is a pure function of the mask.
__global__ void contour_init_label_kernel(
    const uint8_t* mask, int mask_stride, int x, int y, int width, int height,
    signed char* label, int label_stride) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (column > width + 1 || row > height + 1) return;
    signed char value = 0;
    if (column >= 1 && column <= width && row >= 1 && row <= height) {
        const uint8_t* source =
            mask + static_cast<size_t>(y + row - 1) * mask_stride + static_cast<size_t>(x + column - 1);
        value = (*source != 0) ? static_cast<signed char>(1) : static_cast<signed char>(0);
    }
    label[static_cast<size_t>(row) * label_stride + column] = value;
}

// Decides what the raster scan must do at a stop position: a stop is any column where the current
// label differs from the column to its left. Returns true when a border must be opened, with
// *is_hole set to 0 (outer) or 1 (hole). The two else-branches mirror cvFindNextContour's
// resume_scan path, including its lnbd bookkeeping.
__device__ __forceinline__ bool contour_stop_starts_border(
    const signed char* image, int stride, int y, int x, int mode,
    int* lnbd_x, int* lnbd_y, int* is_hole_out) {
    const int prev = static_cast<int>(image[static_cast<size_t>(y) * stride + (x - 1)]);
    const int p = static_cast<int>(image[static_cast<size_t>(y) * stride + x]);
    *is_hole_out = 0;
    if (!(prev == 0 && p == 1)) {
        // Not an outer border. `p != 0 || prev < 1` also rejects a hole start where the left pixel
        // carries the -126 right-bound mark (which is < 1).
        if (p != 0 || prev < 1) {
            if (p & -2) *lnbd_x = x;
            return false;
        }
        *is_hole_out = 1;
    }
    // RETR_EXTERNAL skips hole borders and borders whose left neighbour already carries a label.
    if (mode == VF_CONTOURS_RETR_EXTERNAL &&
        (*is_hole_out != 0 ||
         static_cast<int>(image[static_cast<size_t>(*lnbd_y) * stride + *lnbd_x]) > 0)) {
        if (p & -2) *lnbd_x = x;
        return false;
    }
    return true;
}

// Opens one border at (y, x) and traces it into the discovery-order scratch.
__device__ __forceinline__ void contour_open_border(
    signed char* image, int stride, int y, int x, int is_hole,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* contour_count, int* point_index, int* overflow) {
    const int origin_y = y;
    const int origin_x = x - is_hole;
    if (*contour_count < offset_capacity) {
        offsets[*contour_count] = *point_index;
    } else {
        *overflow = 1;
    }
    contour_fetch(
        image, stride, origin_y, origin_x, is_hole, origin_x - 1, origin_y - 1,
        points, point_capacity, point_index, overflow);
    if (*contour_count + 1 < offset_capacity) {
        offsets[*contour_count + 1] = *point_index;
    } else {
        *overflow = 1;
    }
    *contour_count += 1;
}

// Serial port of the cvStartFindContours_Impl / cvFindNextContour raster scan plus the per-border
// trace. A single thread owns the whole scan, so the marking order is the reference order and no
// synchronization or atomic is involved; that is what makes the operator deterministic.
//
// counts[0] = contour count, counts[1] = point count, counts[2] = overflow flag. The counts are
// reported even when the buffers were too small, so the caller can retry with the exact size.
// This literal row walk is the reference form and stays in charge of RETR_EXTERNAL, where the
// scan's lnbd bookkeeping can be updated by stops that this walk sees and a transition list does
// not. RETR_LIST takes contour_scan_list_kernel instead.
__global__ void contour_scan_kernel(
    signed char* image, int stride, int width, int height, int mode,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* counts) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0) return;
    const int scan_w = width + 1;  // scanner->img_size.width  = W + 2 - 1
    const int scan_h = height + 1; // scanner->img_size.height = H + 2 - 1
    int contour_count = 0;
    int point_index = 0;
    int overflow = 0;

    int x = 1;
    int y = 1;
    int lnbd_x = 0;
    int lnbd_y = 1;
    int prev = static_cast<int>(image[static_cast<size_t>(y) * stride + (x - 1)]);

    while (y < scan_h) {
        int restarted = 0;
        signed char* row = image + static_cast<size_t>(y) * stride;
        while (x < scan_w) {
            while (x < scan_w && static_cast<int>(row[x]) == prev) x += 1;
            if (x >= scan_w) break;
            int is_hole = 0;
            if (contour_stop_starts_border(image, stride, y, x, mode, &lnbd_x, &lnbd_y, &is_hole)) {
                lnbd_x = x - is_hole;
                lnbd_y = y;
                contour_open_border(
                    image, stride, y, x, is_hole, offsets, offset_capacity,
                    points, point_capacity, &contour_count, &point_index, &overflow);
                restarted = 1;
            }
            x += 1;
            prev = static_cast<int>(row[x - 1]);
            if (restarted) break;
        }
        if (restarted) continue;
        lnbd_x = 0;
        lnbd_y = y + 1;
        x = 1;
        prev = 0;
        y += 1;
    }

    counts[0] = contour_count;
    counts[1] = point_index;
    counts[2] = overflow;
}

// Row transition counts of the region's zero-ness: a column where the binary value differs from its
// left neighbour (the padded frame counts as background). Every border this scan can open sits on
// such a column, because both the outer rule (bg -> fg) and the hole rule (fg -> bg) require the
// value change, and marking only rewrites 1 into 2 or -126, which never changes zero-ness.
// One thread per row keeps the pass deterministic; the count is exact, so the list needs no guess.
__global__ void contour_row_transition_counts_kernel(
    const uint8_t* mask, int mask_stride, int x0, int y0, int width, int height,
    int* row_counts) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row > height) return;
    const uint8_t* source = mask + static_cast<size_t>(y0 + row - 1) * mask_stride + x0;
    int previous = 0;
    int count = 0;
    for (int column = 1; column <= width; ++column) {
        const int value = source[column - 1] != 0 ? 1 : 0;
        if (value != previous) count += 1;
        previous = value;
    }
    row_counts[row - 1] = count;
}

// Fills the raster-ordered list of padded label indices, one thread per row, each row writing its
// own segment in ascending column order.
__global__ void contour_fill_transitions_kernel(
    const uint8_t* mask, int mask_stride, int x0, int y0, int width, int height,
    const int* row_start, int stride, int* transitions) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (row > height) return;
    const uint8_t* source = mask + static_cast<size_t>(y0 + row - 1) * mask_stride + x0;
    int previous = 0;
    int index = row_start[row];
    const int base = row * stride;
    for (int column = 1; column <= width; ++column) {
        const int value = source[column - 1] != 0 ? 1 : 0;
        if (value != previous) {
            transitions[index] = base + column;
            index += 1;
        }
        previous = value;
    }
}

// RETR_LIST scan over the precomputed transition list. Iterating only the value changes is exact:
// a stop whose left neighbour keeps the same zero-ness can only take the harmless resume_scan
// branch (it updates prev, which this kernel re-reads from the image at every stop, and lnbd_x,
// which RETR_LIST never reads). The literal row walk and this walk therefore open the same borders
// in the same order, while this one touches memory only where a decision can happen.
__global__ void contour_scan_list_kernel(
    signed char* image, int stride, int height,
    const int32_t* transitions, const int32_t* row_start,
    int32_t* offsets, int offset_capacity,
    int32_t* points, int point_capacity,
    int* counts) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || threadIdx.x != 0 || threadIdx.y != 0) return;
    int contour_count = 0;
    int point_index = 0;
    int overflow = 0;
    int lnbd_x = 0;
    int lnbd_y = 1;

    for (int y = 1; y <= height; ++y) {
        const int end = row_start[y + 1];
        for (int index = row_start[y]; index < end; ++index) {
            const int position = transitions[index];
            const int x = position - y * stride;
            int is_hole = 0;
            if (!contour_stop_starts_border(
                    image, stride, y, x, VF_CONTOURS_RETR_LIST, &lnbd_x, &lnbd_y, &is_hole)) {
                continue;
            }
            lnbd_x = x - is_hole;
            lnbd_y = y;
            contour_open_border(
                image, stride, y, x, is_hole, offsets, offset_capacity,
                points, point_capacity, &contour_count, &point_index, &overflow);
        }
    }

    counts[0] = contour_count;
    counts[1] = point_index;
    counts[2] = overflow;
}

// Reverses the discovery-order scratch into the OpenCV order. Each contour is copied by one thread
// that also derives its output offset from the monotonic offset table, so the result is a pure
// function of the scratch.
__global__ void contour_reverse_kernel(
    const int32_t* offsets, const int32_t* points, int contour_count, int point_count,
    int32_t* out_offsets, int32_t* out_points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= contour_count) return;
    const int source = contour_count - 1 - index;
    const int source_start = offsets[source];
    const int source_end = offsets[source + 1];
    const int target_start = point_count - offsets[contour_count - index];
    out_offsets[index] = target_start;
    if (index == 0) out_offsets[contour_count] = point_count;
    for (int point = source_start; point < source_end; ++point) {
        const int target = target_start + (point - source_start);
        out_points[static_cast<size_t>(target) * 2] = points[static_cast<size_t>(point) * 2];
        out_points[static_cast<size_t>(target) * 2 + 1] = points[static_cast<size_t>(point) * 2 + 1];
    }
}
}

static int execute_linear_plan_device(
    NativePlan* compiled,
    uint8_t* current,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    PersistentContext* context = compiled->context;
    int width = compiled->width;
    int height = compiled->height;
    int channels = compiled->input_channels;
    size_t area_resize_index = 0;
    for (const VfPlanOperatorV1& op : compiled->operators) {
        uint8_t* next = current == context->u8[1] ? context->u8[2] : context->u8[1];
        switch (op.kind) {
            case VF_PLAN_GRAY:
                if (channels == 3) {
                    bgr_gray_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        current, next, width, height);
                    current = next;
                    channels = 1;
                }
                break;
            case VF_PLAN_RESIZE_AREA: {
                const int target_width = op.int_params[0];
                const int target_height = op.int_params[1];
                if (area_resize_index >= compiled->area_resizes.size()) return VF_CUDA_INTERNAL_ERROR;
                launch_area_resize(
                    *compiled->area_resizes[area_resize_index++], current, next, width,
                    target_width, target_height, context->stream);
                current = next;
                width = target_width;
                height = target_height;
                break;
            }
            case VF_PLAN_GAUSSIAN: {
                context->timing_has_gaussian = true;
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_START], context->stream);
                int radius = 0;
                int result = prepare_gaussian_weights(
                    op.int_params[0], &radius, context->stream);
                if (result != VF_CUDA_OK) return result;
                launch_gaussian(
                    current, context->gaussian_buffer, next, width, height, channels, radius,
                    context->stream);
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_END], context->stream);
                current = next;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_START], context->stream);
                threshold_kernel<<<(width * height + 255) / 256, 256, 0, context->stream>>>(
                    current, next, width * height, op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_END], context->stream);
                current = next;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_START], context->stream);
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                int result = adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                             &padded_height, &padded_count);
                if (result != VF_CUDA_OK) return result;
                replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, context->u8[3], width, height, padded_width, padded_height, radius);
                row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, context->stream>>>(
                    context->u8[3], context->u64[0], padded_width, padded_height);
                dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
                dim3 transpose_grid(
                    (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
                    (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
                transpose_u64_kernel<<<transpose_grid, transpose_block, 0, context->stream>>>(
                    context->u64[0], context->u64[1], padded_width, padded_height);
                row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, context->stream>>>(
                    context->u64[1], padded_height, padded_width);
                adaptive_integral_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    current, context->u64[1], next, width, height, padded_height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_END], context->stream);
                current = next;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_START], context->stream);
                const int operation = op.int_params[0];
                const int kernel = op.int_params[1];
                const int iterations = op.int_params[2];
                const int passes = (operation == VF_MORPH_OPEN || operation == VF_MORPH_CLOSE)
                    ? iterations * 2 : iterations;
                uint8_t* source_buffer = current;
                uint8_t* next = current == context->u8[1] ? context->u8[2] : context->u8[1];
                uint8_t* destination = passes % 2 == 1 ? next : context->u8[4];
                for (int pass = 0; pass < passes; ++pass) {
                    int dilate = operation == VF_MORPH_DILATE;
                    if (operation == VF_MORPH_OPEN) dilate = pass >= iterations;
                    if (operation == VF_MORPH_CLOSE) dilate = pass < iterations;
                    launch_morph_pass(
                        source_buffer, destination, width, height, channels,
                        kernel / 2, dilate, context->stream);
                    source_buffer = destination;
                    destination = destination == next ? context->u8[4] : next;
                }
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_END], context->stream);
                current = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cudaEventRecord(context->timing_events[TIMING_AFTER_KERNEL], context->stream);
    const size_t output_row_bytes = static_cast<size_t>(width) * dst_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        dst, dst_stride, current, output_row_bytes, output_row_bytes, height,
        cudaMemcpyDeviceToHost, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_OUTPUT], context->stream);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(context->stream);
    context->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(context);
    return result;
}

static int execute_dag_plan_device(
    NativeDagPlan* compiled,
    uint8_t* root,
    const VfDagOutputV1* outputs,
    int output_count) {
    PersistentContext* context = compiled->context;
    const int width = compiled->width;
    const int height = compiled->height;
    const size_t pixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t*> values(compiled->operators.size(), nullptr);
    for (size_t index = 0; index < compiled->operators.size(); ++index) {
        const VfPlanOperatorV1& op = compiled->operators[index];
        uint8_t* input = op.input_node == VF_PLAN_INPUT_NODE ? root : values[op.input_node];
        uint8_t* output = context->dag_u8[index];
        int channels = op.input_node == VF_PLAN_INPUT_NODE
            ? compiled->input_channels : compiled->node_channels[op.input_node];
        switch (op.kind) {
            case VF_PLAN_GRAY:
                if (channels == 3) {
                    bgr_gray_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                        input, output, width, height);
                    values[index] = output;
                } else {
                    values[index] = input;
                }
                break;
            case VF_PLAN_GAUSSIAN: {
                context->timing_has_gaussian = true;
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_START], context->stream);
                int radius = 0;
                int result = prepare_gaussian_weights(
                    op.int_params[0], &radius, context->stream);
                if (result != VF_CUDA_OK) return result;
                launch_gaussian(
                    input, context->gaussian_buffer, output, width, height, channels, radius,
                    context->stream);
                cudaEventRecord(context->timing_events[TIMING_GAUSSIAN_END], context->stream);
                values[index] = output;
                break;
            }
            case VF_PLAN_THRESHOLD:
                context->timing_has_threshold = true;
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_START], context->stream);
                threshold_kernel<<<(static_cast<int>(pixels) + 255) / 256, 256, 0, context->stream>>>(
                    input, output, static_cast<int>(pixels), op.int_params[0],
                    op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_THRESHOLD_END], context->stream);
                values[index] = output;
                break;
            case VF_PLAN_ADAPTIVE_MEAN: {
                context->timing_has_adaptive = true;
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_START], context->stream);
                int radius = 0, padded_width = 0, padded_height = 0;
                size_t padded_count = 0;
                int result = adaptive_layout(width, height, op.int_params[0], &radius, &padded_width,
                                             &padded_height, &padded_count);
                if (result != VF_CUDA_OK) return result;
                replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    input, context->u8[3], width, height, padded_width, padded_height, radius);
                row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, context->stream>>>(
                    context->u8[3], context->u64[0], padded_width, padded_height);
                dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
                dim3 transpose_grid(
                    (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
                    (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
                transpose_u64_kernel<<<transpose_grid, transpose_block, 0, context->stream>>>(
                    context->u64[0], context->u64[1], padded_width, padded_height);
                row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, context->stream>>>(
                    context->u64[1], padded_height, padded_width);
                adaptive_integral_kernel<<<grid2d(width, height), dim3(BLOCK_X, BLOCK_Y), 0, context->stream>>>(
                    input, context->u64[1], output, width, height, padded_height,
                    op.int_params[0], op.float_params[0], op.int_params[1], op.int_params[2]);
                cudaEventRecord(context->timing_events[TIMING_ADAPTIVE_END], context->stream);
                values[index] = output;
                break;
            }
            case VF_PLAN_MORPHOLOGY: {
                context->timing_has_morphology = true;
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_START], context->stream);
                const int operation = op.int_params[0];
                const int iterations = op.int_params[2];
                const int passes = (operation == VF_MORPH_OPEN || operation == VF_MORPH_CLOSE)
                    ? iterations * 2 : iterations;
                uint8_t* source_buffer = input;
                uint8_t* destination = passes % 2 == 1 ? output : context->u8[4];
                for (int pass = 0; pass < passes; ++pass) {
                    int dilate = operation == VF_MORPH_DILATE;
                    if (operation == VF_MORPH_OPEN) dilate = pass >= iterations;
                    if (operation == VF_MORPH_CLOSE) dilate = pass < iterations;
                    launch_morph_pass(
                        source_buffer, destination, width, height, channels,
                        op.int_params[1] / 2, dilate, context->stream);
                    source_buffer = destination;
                    destination = destination == output ? context->u8[4] : output;
                }
                cudaEventRecord(context->timing_events[TIMING_MORPHOLOGY_END], context->stream);
                values[index] = source_buffer;
                break;
            }
            default:
                return VF_CUDA_UNSUPPORTED;
        }
    }
    int result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cudaEventRecord(context->timing_events[TIMING_AFTER_KERNEL], context->stream);
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        size_t row_bytes = static_cast<size_t>(width) * compiled->node_channels[node];
        cudaError_t error = cudaMemcpy2DAsync(
            outputs[index].data, outputs[index].stride, values[node], row_bytes,
            row_bytes, height, cudaMemcpyDeviceToHost, context->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    cudaEventRecord(context->timing_events[TIMING_AFTER_OUTPUT], context->stream);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(context->stream);
    context->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(context);
    return result;
}

VF_CUDA_API int vf_gpu_abi_version() { return VF_CUDA_ABI_VERSION; }

VF_CUDA_API int vf_gpu_device_count() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0; }

VF_CUDA_API int vf_gpu_compute_capability() {
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? prop.major * 10 + prop.minor : 0;
}

VF_CUDA_API int vf_gpu_device_name(char* output, int capacity) {
    if (!output || capacity <= 0) return 1;
    cudaDeviceProp prop{}; cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) return cuda_result(error);
    strncpy_s(output, capacity, prop.name, _TRUNCATE); return 0;
}

VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity) {
    if (!output || capacity <= 0) return VF_CUDA_INVALID_ARGUMENT;
    const char* message = "Unknown VisionFlow CUDA error";
    switch (error_code) {
        case VF_CUDA_OK: message = "Success"; break;
        case VF_CUDA_INVALID_ARGUMENT: message = "Invalid argument"; break;
        case VF_CUDA_ALLOCATION_FAILED: message = "Device allocation failed"; break;
        case VF_CUDA_COPY_FAILED: message = "Host/device copy failed"; break;
        case VF_CUDA_KERNEL_FAILED: message = "CUDA kernel failed"; break;
        case VF_CUDA_DEVICE_UNAVAILABLE: message = "CUDA device unavailable"; break;
        case VF_CUDA_ABI_MISMATCH: message = "CUDA DLL ABI mismatch"; break;
        case VF_CUDA_INTERNAL_ERROR: message = "Internal CUDA DLL error"; break;
        case VF_CUDA_UNSUPPORTED: message = "Requested CUDA operation is unsupported"; break;
        default:
            if (error_code >= VF_CUDA_RUNTIME_ERROR_BASE) {
                message = cudaGetErrorString(static_cast<cudaError_t>(error_code - VF_CUDA_RUNTIME_ERROR_BASE));
            }
            break;
    }
    strncpy_s(output, capacity, message, _TRUNCATE);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_gpu_memory_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    size_t free_value = 0;
    size_t total_value = 0;
    cudaError_t error = cudaMemGetInfo(&free_value, &total_value);
    if (error != cudaSuccess) return cuda_result(error);
    *free_bytes = static_cast<uint64_t>(free_value);
    *total_bytes = static_cast<uint64_t>(total_value);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_create(void** context) {
    if (context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *context = nullptr;
    auto started = std::chrono::steady_clock::now();
    PersistentContext* created = new (std::nothrow) PersistentContext();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    if (created->initialization_error != cudaSuccess) {
        int result = cuda_result(created->initialization_error);
        delete created;
        return result;
    }
    created->last_timings.context_create_ms = elapsed_host_ms(started);
    *context = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_last_timings(void* context, VfCudaTimingsV1* timings) {
    if (context == nullptr || timings == nullptr ||
        timings->struct_size != sizeof(VfCudaTimingsV1) || timings->version != 1) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *timings = static_cast<PersistentContext*>(context)->last_timings;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_destroy(void* context) {
    delete static_cast<PersistentContext*>(context);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_stats(
    void* context,
    uint64_t* reserved_bytes,
    uint64_t* allocation_count) {
    if (context == nullptr || reserved_bytes == nullptr || allocation_count == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    uint64_t bytes = 0;
    for (size_t capacity : persistent->u8_capacity) bytes += static_cast<uint64_t>(capacity);
    bytes += static_cast<uint64_t>(persistent->gaussian_capacity) * sizeof(uint32_t);
    for (size_t capacity : persistent->u64_capacity) {
        bytes += static_cast<uint64_t>(capacity) * sizeof(unsigned long long);
    }
    for (size_t capacity : persistent->dag_u8_capacity) bytes += static_cast<uint64_t>(capacity);
    bytes += static_cast<uint64_t>(persistent->resident_capacity);
    *reserved_bytes = bytes;
    *allocation_count = persistent->allocation_count;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_context_upload_u8(
    void* context,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint64_t* generation) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || generation == nullptr ||
        (src_channels != 1 && src_channels != 3) ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    reset_timing(persistent, true);
    size_t row_bytes = static_cast<size_t>(width) * src_channels;
    auto allocation_started = std::chrono::steady_clock::now();
    int result = reserve_device(
        &persistent->resident_u8, &persistent->resident_capacity,
        row_bytes * static_cast<size_t>(height), &persistent->allocation_count);
    persistent->last_timings.allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) return result;
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->resident_u8, row_bytes, src, src_stride, row_bytes, height,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_INPUT], persistent->stream);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_KERNEL], persistent->stream);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_OUTPUT], persistent->stream);
    auto synchronize_started = std::chrono::steady_clock::now();
    result = visionflow_cuda::stream_result(persistent->stream);
    persistent->last_timings.synchronize_ms = elapsed_host_ms(synchronize_started);
    if (result == VF_CUDA_OK) finalize_timing(persistent);
    if (result != VF_CUDA_OK) return result;
    persistent->resident_width = width;
    persistent->resident_height = height;
    persistent->resident_channels = src_channels;
    ++persistent->resident_generation;
    if (persistent->resident_generation == 0) ++persistent->resident_generation;
    *generation = persistent->resident_generation;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_roi_batch_create(
    void* context,
    uint64_t generation,
    const VfRoiV1* rois,
    int roi_count,
    void** batch) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || batch == nullptr || rois == nullptr || roi_count <= 0 ||
        roi_count > 65535 || generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *batch = nullptr;
    const int width = rois[0].width;
    const int height = rois[0].height;
    if (width <= 0 || height <= 0 || width > persistent->resident_width ||
        height > persistent->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < roi_count; ++index) {
        const VfRoiV1& roi = rois[index];
        if (roi.struct_size != sizeof(VfRoiV1) || roi.width != width || roi.height != height ||
            roi.x < 0 || roi.y < 0 || roi.x > persistent->resident_width - width ||
            roi.y > persistent->resident_height - height) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    size_t roi_bytes = static_cast<size_t>(width) * height * persistent->resident_channels;
    if (roi_bytes == 0 || static_cast<size_t>(roi_count) > SIZE_MAX / roi_bytes ||
        static_cast<size_t>(roi_count) > SIZE_MAX / sizeof(VfRoiV1)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    NativeRoiBatch* created = new (std::nothrow) NativeRoiBatch();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = persistent;
    created->count = roi_count;
    created->width = width;
    created->height = height;
    created->channels = persistent->resident_channels;
    cudaError_t error = cudaMalloc(&created->data, roi_bytes * roi_count);
    if (error == cudaSuccess) {
        error = cudaMalloc(&created->device_rois, sizeof(VfRoiV1) * roi_count);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            created->device_rois, rois, sizeof(VfRoiV1) * roi_count,
            cudaMemcpyHostToDevice, persistent->stream);
    }
    if (error != cudaSuccess) {
        delete created;
        return cuda_result(error);
    }
    dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y,
        roi_count);
    gather_roi_batch_kernel<<<grid, dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->resident_u8, persistent->resident_width, persistent->resident_channels,
        created->device_rois, created->data, width, height);
    int result = visionflow_cuda::kernel_launch_result();
    if (result == VF_CUDA_OK) result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *batch = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_roi_batch_info(
    void* batch,
    int* roi_count,
    int* width,
    int* height,
    int* channels) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr || roi_count == nullptr || width == nullptr || height == nullptr ||
        channels == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    *roi_count = native->count;
    *width = native->width;
    *height = native->height;
    *channels = native->channels;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_roi_batch_download_u8(
    void* batch,
    int roi_index,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr || native->context == nullptr || roi_index < 0 ||
        roi_index >= native->count || dst_channels != native->channels ||
        !visionflow_cuda::valid_image(
            dst, native->width, native->height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t row_bytes = static_cast<size_t>(native->width) * native->channels;
    size_t roi_bytes = row_bytes * native->height;
    cudaError_t error = cudaMemcpy2DAsync(
        dst, dst_stride, native->data + static_cast<size_t>(roi_index) * roi_bytes,
        row_bytes, row_bytes, native->height, cudaMemcpyDeviceToHost,
        native->context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(native->context->stream);
}

VF_CUDA_API int vf_roi_batch_destroy(void* batch) {
    NativeRoiBatch* native = static_cast<NativeRoiBatch*>(batch);
    if (native == nullptr) return VF_CUDA_OK;
    PersistentContext* context = native->context;
    auto started = std::chrono::steady_clock::now();
    delete native;
    if (context != nullptr) context->last_timings.free_ms = elapsed_host_ms(started);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_query(
    const VfPlanDescV1* desc,
    int width,
    int height,
    char* reason,
    int reason_capacity) {
    return validate_plan_desc(
        desc, width, height, nullptr, nullptr, nullptr, reason, reason_capacity);
}

VF_CUDA_API int vf_plan_create(
    void* context,
    const VfPlanDescV1* desc,
    int width,
    int height,
    void** plan) {
    if (context == nullptr || plan == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *plan = nullptr;
    int output_channels = 0;
    int output_width = 0;
    int output_height = 0;
    int result = validate_plan_desc(
        desc, width, height, &output_channels, &output_width, &output_height, nullptr, 0);
    if (result != VF_CUDA_OK) return result;

    NativePlan* created = new (std::nothrow) NativePlan();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = static_cast<PersistentContext*>(context);
    created->width = width;
    created->height = height;
    created->output_width = output_width;
    created->output_height = output_height;
    created->input_channels = desc->input_channels;
    created->output_channels = output_channels;
    try {
        created->operators.assign(desc->operators, desc->operators + desc->operator_count);
    } catch (const std::bad_alloc&) {
        delete created;
        return VF_CUDA_ALLOCATION_FAILED;
    }
    auto allocation_started = std::chrono::steady_clock::now();
    result = reserve_plan_buffers(created->context, *created);
    int current_width = width;
    int current_height = height;
    for (const VfPlanOperatorV1& op : created->operators) {
        if (result != VF_CUDA_OK) break;
        if (op.kind != VF_PLAN_RESIZE_AREA) continue;
        std::unique_ptr<AreaResizeTables> tables(new (std::nothrow) AreaResizeTables());
        if (!tables) {
            result = VF_CUDA_ALLOCATION_FAILED;
            break;
        }
        result = prepare_area_resize(
            current_width, current_height, op.int_params[0], op.int_params[1], tables.get(),
            &created->context->allocation_count);
        current_width = op.int_params[0];
        current_height = op.int_params[1];
        if (result == VF_CUDA_OK) {
            try {
                created->area_resizes.push_back(std::move(tables));
            } catch (const std::bad_alloc&) {
                result = VF_CUDA_ALLOCATION_FAILED;
            }
        }
    }
    created->context->pending_allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *plan = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_execute(
    void* plan,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativePlan* compiled = static_cast<NativePlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || width != compiled->width ||
        height != compiled->height || src_channels != compiled->input_channels ||
        dst_channels != compiled->output_channels ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels) ||
        !visionflow_cuda::valid_image(
            dst, compiled->output_width, compiled->output_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* context = compiled->context;
    reset_timing(context, true);
    const size_t source_row_bytes = static_cast<size_t>(width) * src_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], source_row_bytes, src, src_stride, source_row_bytes, height,
        cudaMemcpyHostToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);

    return execute_linear_plan_device(
        compiled, context->u8[0], dst, dst_stride, dst_channels);
}

VF_CUDA_API int vf_plan_destroy(void* plan) {
    delete static_cast<NativePlan*>(plan);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_plan_execute_roi(
    void* plan,
    uint64_t generation,
    int x,
    int y,
    uint8_t* dst,
    int dst_stride,
    int dst_channels) {
    NativePlan* compiled = static_cast<NativePlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    PersistentContext* context = compiled->context;
    if (generation == 0 || generation != context->resident_generation ||
        context->resident_channels != compiled->input_channels || x < 0 || y < 0 ||
        x + compiled->width > context->resident_width ||
        y + compiled->height > context->resident_height ||
        dst_channels != compiled->output_channels ||
        !visionflow_cuda::valid_image(
            dst, compiled->output_width, compiled->output_height, dst_stride, dst_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    reset_timing(context, false);
    size_t resident_pitch = static_cast<size_t>(context->resident_width) * context->resident_channels;
    size_t roi_row_bytes = static_cast<size_t>(compiled->width) * compiled->input_channels;
    const uint8_t* source = context->resident_u8 +
        static_cast<size_t>(y) * resident_pitch + static_cast<size_t>(x) * compiled->input_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], roi_row_bytes, source, resident_pitch, roi_row_bytes, compiled->height,
        cudaMemcpyDeviceToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);
    return execute_linear_plan_device(
        compiled, context->u8[0], dst, dst_stride, dst_channels);
}

VF_CUDA_API int vf_dag_plan_query(
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    char* reason,
    int reason_capacity) {
    return validate_dag_plan_desc(desc, width, height, nullptr, reason, reason_capacity);
}

VF_CUDA_API int vf_dag_plan_create(
    void* context,
    const VfDagPlanDescV1* desc,
    int width,
    int height,
    void** plan) {
    if (context == nullptr || plan == nullptr) return VF_CUDA_INVALID_ARGUMENT;
    *plan = nullptr;
    std::vector<int> node_channels;
    int result = validate_dag_plan_desc(desc, width, height, &node_channels, nullptr, 0);
    if (result != VF_CUDA_OK) return result;
    NativeDagPlan* created = new (std::nothrow) NativeDagPlan();
    if (created == nullptr) return VF_CUDA_ALLOCATION_FAILED;
    created->context = static_cast<PersistentContext*>(context);
    created->width = width;
    created->height = height;
    created->input_channels = desc->input_channels;
    try {
        created->operators.assign(desc->operators, desc->operators + desc->operator_count);
        created->node_channels = std::move(node_channels);
        created->output_nodes.assign(desc->output_nodes, desc->output_nodes + desc->output_count);
    } catch (const std::bad_alloc&) {
        delete created;
        return VF_CUDA_ALLOCATION_FAILED;
    }
    auto allocation_started = std::chrono::steady_clock::now();
    result = reserve_dag_plan_buffers(created->context, *created);
    created->context->pending_allocation_ms += elapsed_host_ms(allocation_started);
    if (result != VF_CUDA_OK) {
        delete created;
        return result;
    }
    *plan = created;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_dag_plan_execute(
    void* plan,
    const uint8_t* src,
    int width,
    int height,
    int src_stride,
    int src_channels,
    const VfDagOutputV1* outputs,
    int output_count) {
    NativeDagPlan* compiled = static_cast<NativeDagPlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || width != compiled->width ||
        height != compiled->height || src_channels != compiled->input_channels ||
        output_count != static_cast<int>(compiled->output_nodes.size()) || outputs == nullptr ||
        !visionflow_cuda::valid_image(src, width, height, src_stride, src_channels)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        if (outputs[index].struct_size != sizeof(VfDagOutputV1) || outputs[index].node != node ||
            outputs[index].channels != compiled->node_channels[node] ||
            !visionflow_cuda::valid_image(
                outputs[index].data, width, height, outputs[index].stride, outputs[index].channels)) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    PersistentContext* context = compiled->context;
    reset_timing(context, true);
    const size_t source_row_bytes = static_cast<size_t>(width) * src_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], source_row_bytes, src, src_stride, source_row_bytes, height,
        cudaMemcpyHostToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);

    return execute_dag_plan_device(
        compiled, context->u8[0], outputs, output_count);
}

VF_CUDA_API int vf_dag_plan_destroy(void* plan) {
    delete static_cast<NativeDagPlan*>(plan);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_dag_plan_execute_roi(
    void* plan,
    uint64_t generation,
    int x,
    int y,
    const VfDagOutputV1* outputs,
    int output_count) {
    NativeDagPlan* compiled = static_cast<NativeDagPlan*>(plan);
    if (compiled == nullptr || compiled->context == nullptr || outputs == nullptr ||
        output_count != static_cast<int>(compiled->output_nodes.size())) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    PersistentContext* context = compiled->context;
    if (generation == 0 || generation != context->resident_generation ||
        context->resident_channels != compiled->input_channels || x < 0 || y < 0 ||
        x + compiled->width > context->resident_width ||
        y + compiled->height > context->resident_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    for (int index = 0; index < output_count; ++index) {
        int node = compiled->output_nodes[index];
        if (outputs[index].struct_size != sizeof(VfDagOutputV1) || outputs[index].node != node ||
            outputs[index].channels != compiled->node_channels[node] ||
            !visionflow_cuda::valid_image(
                outputs[index].data, compiled->width, compiled->height,
                outputs[index].stride, outputs[index].channels)) {
            return VF_CUDA_INVALID_ARGUMENT;
        }
    }
    reset_timing(context, false);
    size_t resident_pitch = static_cast<size_t>(context->resident_width) * context->resident_channels;
    size_t roi_row_bytes = static_cast<size_t>(compiled->width) * compiled->input_channels;
    const uint8_t* source = context->resident_u8 +
        static_cast<size_t>(y) * resident_pitch + static_cast<size_t>(x) * compiled->input_channels;
    cudaError_t error = cudaMemcpy2DAsync(
        context->u8[0], roi_row_bytes, source, resident_pitch, roi_row_bytes, compiled->height,
        cudaMemcpyDeviceToDevice, context->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(context->timing_events[TIMING_AFTER_INPUT], context->stream);
    return execute_dag_plan_device(
        compiled, context->u8[0], outputs, output_count);
}

VF_CUDA_API int vf_bgr_to_gray_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 1) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_gray_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_bgr_to_rgb_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 3) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * 3);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_rgb_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 3, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_crop_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int x,int y,int cw,int ch) {
    if (sc != dc || (sc != 1 && sc != 3) || x < 0 || y < 0 || cw <= 0 || ch <= 0 || x + cw > w || y + ch > h) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(cw) * ch * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    crop_kernel<<<grid2d(cw, ch), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, x, y, cw, ch, sc);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, cw, ch, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_resize_gray_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int dw,int dh) {
    if (sc != 1 || dc != 1 || dw <= 0 || dh <= 0) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(dw) * dh);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    if (dw <= w && dh <= h) {
        AreaResizeTables tables;
        result = prepare_area_resize(w, h, dw, dh, &tables, nullptr);
        if (result == VF_CUDA_OK) {
            launch_area_resize(tables, ds, dd, w, dw, dh);
            result = visionflow_cuda::kernel_result();
        }
    } else {
        resize_gray_kernel<<<grid2d(dw, dh), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h, dw, dh);
        result = visionflow_cuda::kernel_result();
    }
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, dw, dh, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_gaussian_blur_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int kernel) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || kernel > MAX_GAUSSIAN_KERNEL) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    uint32_t* intermediate = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    cudaError_t error = cudaMalloc(
        &intermediate, static_cast<size_t>(w) * h * sc * sizeof(uint32_t));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }

    int radius = 0;
    result = prepare_gaussian_weights(kernel, &radius);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(intermediate);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return result;
    }
    launch_gaussian(ds, intermediate, dd, w, h, sc, radius);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(intermediate);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_threshold_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int threshold,int max_value,int invert) {
    if (sc != 1 || dc != 1 || threshold < 0 || threshold > 255 || max_value < 0 || max_value > 255) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    int count = w * h;
    threshold_kernel<<<(count + 255) / 256, 256>>>(ds, dd, count, threshold, max_value, invert);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_adaptive_mean_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int block,float c,int max_value,int invert) {
    if (w <= 0 || h <= 0 || sc != 1 || dc != 1 || block < 3 || block % 2 == 0 ||
        max_value < 0 || max_value > 255 || !std::isfinite(c)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int radius = 0, padded_width = 0, padded_height = 0;
    size_t padded_count = 0;
    int result = adaptive_layout(
        w, h, block, &radius, &padded_width, &padded_height, &padded_count);
    if (result != VF_CUDA_OK) return result;
    uint8_t *ds = nullptr, *dd = nullptr;
    uint8_t* padded = nullptr;
    unsigned long long* row_prefix = nullptr;
    unsigned long long* integral_transposed = nullptr;
    result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    result = visionflow_cuda::allocate_bytes(&padded, padded_count);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return result;
    }
    cudaError_t error = cudaMalloc(&row_prefix, padded_count * sizeof(unsigned long long));
    if (error == cudaSuccess) error = cudaMalloc(&integral_transposed, padded_count * sizeof(unsigned long long));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(integral_transposed);
        visionflow_cuda::free_device(row_prefix);
        visionflow_cuda::free_device(padded);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }
    replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, padded, w, h, padded_width, padded_height, radius);
    row_prefix_u8_kernel<<<padded_height, SCAN_THREADS>>>(padded, row_prefix, padded_width, padded_height);
    dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
    dim3 transpose_grid(
        (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
    transpose_u64_kernel<<<transpose_grid, transpose_block>>>(
        row_prefix, integral_transposed, padded_width, padded_height);
    row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS>>>(
        integral_transposed, padded_height, padded_width);
    adaptive_integral_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, integral_transposed, dd, w, h, padded_height, block, c, max_value, invert);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(integral_transposed);
    visionflow_cuda::free_device(row_prefix);
    visionflow_cuda::free_device(padded);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_preprocess_401_2_u8(
    void* context,
    const uint8_t* src,
    int w,
    int h,
    int stride,
    int sc,
    uint8_t* dst,
    int dstride,
    int gaussian_kernel,
    int adaptive_block,
    float adaptive_c,
    int max_value,
    int invert) {
    if (context == nullptr || w <= 0 || h <= 0 || (sc != 1 && sc != 3) ||
        w > INT_MAX / sc || !visionflow_cuda::valid_image(src, w, h, stride, sc) ||
        !visionflow_cuda::valid_image(dst, w, h, dstride, 1) ||
        max_value < 0 || max_value > 255 || !std::isfinite(adaptive_c)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (static_cast<size_t>(w) > SIZE_MAX / static_cast<size_t>(h)) return VF_CUDA_INVALID_ARGUMENT;
    size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (pixel_count > SIZE_MAX / static_cast<size_t>(sc)) return VF_CUDA_INVALID_ARGUMENT;
    size_t source_count = pixel_count * static_cast<size_t>(sc);

    int radius = 0;
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    int result = prepare_gaussian_weights(
        gaussian_kernel, &radius, persistent->stream);
    if (result != VF_CUDA_OK) return result;
    int adaptive_radius = 0, padded_width = 0, padded_height = 0;
    size_t padded_count = 0;
    result = adaptive_layout(
        w,
        h,
        adaptive_block,
        &adaptive_radius,
        &padded_width,
        &padded_height,
        &padded_count);
    if (result != VF_CUDA_OK) return result;

    result = reserve_device(
        &persistent->u8[0], &persistent->u8_capacity[0], source_count, &persistent->allocation_count);
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u8[1], &persistent->u8_capacity[1], pixel_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u8[2], &persistent->u8_capacity[2], pixel_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u8[3], &persistent->u8_capacity[3], padded_count, &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->gaussian_buffer,
            &persistent->gaussian_capacity,
            pixel_count,
            &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u64[0],
            &persistent->u64_capacity[0],
            padded_count,
            &persistent->allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = reserve_device(
            &persistent->u64[1],
            &persistent->u64_capacity[1],
            padded_count,
            &persistent->allocation_count);
    }
    if (result != VF_CUDA_OK) return result;

    size_t source_row_bytes = static_cast<size_t>(w) * static_cast<size_t>(sc);
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->u8[0],
        source_row_bytes,
        src,
        stride,
        source_row_bytes,
        h,
        cudaMemcpyHostToDevice,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);

    uint8_t* gray = persistent->u8[0];
    if (sc == 3) {
        gray = persistent->u8[1];
        bgr_gray_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            persistent->u8[0], gray, w, h);
    }
    gaussian_horizontal_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray, persistent->gaussian_buffer, w, h, 1, radius);
    gaussian_vertical_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        persistent->gaussian_buffer, gray, w, h, 1, radius);
    replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray,
        persistent->u8[3],
        w,
        h,
        padded_width,
        padded_height,
        adaptive_radius);
    row_prefix_u8_kernel<<<padded_height, SCAN_THREADS, 0, persistent->stream>>>(
        persistent->u8[3], persistent->u64[0], padded_width, padded_height);
    dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
    dim3 transpose_grid(
        (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
    transpose_u64_kernel<<<transpose_grid, transpose_block, 0, persistent->stream>>>(
        persistent->u64[0], persistent->u64[1], padded_width, padded_height);
    row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS, 0, persistent->stream>>>(
        persistent->u64[1], padded_height, padded_width);
    adaptive_integral_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
        gray,
        persistent->u64[1],
        persistent->u8[2],
        w,
        h,
        padded_height,
        adaptive_block,
        adaptive_c,
        max_value,
        invert);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    error = cudaMemcpy2DAsync(
        dst,
        dstride,
        persistent->u8[2],
        static_cast<size_t>(w),
        static_cast<size_t>(w),
        h,
        cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

VF_CUDA_API int vf_morphology_rect_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int operation,int kernel,int iterations) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || iterations < 1 || operation < VF_MORPH_OPEN || operation > VF_MORPH_ERODE) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *a = nullptr, *b = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &a);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&b, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(a); return result; }
    auto pass = [&](int dilate) {
        launch_morph_pass(a, b, w, h, sc, kernel / 2, dilate);
        std::swap(a, b);
    };
    if (operation == VF_MORPH_OPEN) {
        for (int i = 0; i < iterations; ++i) pass(0);
        for (int i = 0; i < iterations; ++i) pass(1);
    } else if (operation == VF_MORPH_CLOSE) {
        for (int i = 0; i < iterations; ++i) pass(1);
        for (int i = 0; i < iterations; ++i) pass(0);
    } else {
        for (int i = 0; i < iterations; ++i) pass(operation == VF_MORPH_DILATE);
    }
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, a);
    else visionflow_cuda::free_device(a);
    visionflow_cuda::free_device(b);
    return result;
}

VF_CUDA_API int vf_match_template_gray_u8(
    void* context,
    uint64_t generation,
    int search_x, int search_y, int search_width, int search_height,
    const uint8_t* templ, int template_width, int template_height,
    int* out_match, float* out_score) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || templ == nullptr || out_match == nullptr || out_score == nullptr ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (search_x < 0 || search_y < 0 || search_width <= 0 || search_height <= 0 ||
        search_x > persistent->resident_width - search_width ||
        search_y > persistent->resident_height - search_height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (template_width <= 0 || template_height <= 0 ||
        template_width > search_width || template_height > search_height ||
        static_cast<long long>(template_width) * template_height > INT_MAX) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int output_width = search_width - template_width + 1;
    const int output_height = search_height - template_height + 1;
    if (output_width <= 0 || output_height <= 0) return VF_CUDA_INVALID_ARGUMENT;

    const size_t template_bytes = static_cast<size_t>(template_width) * template_height;
    // The prefix planes are indexed by output_width * roi_height, and the context caches them with
    // grow-only semantics, so reserve the ROI area: it bounds every shape this call can index and
    // keeps a later, wider search from reading past an earlier smaller allocation.
    size_t plane_elements = static_cast<size_t>(search_width) * search_height;
    if (plane_elements < static_cast<size_t>(output_width) * output_height) {
        plane_elements = static_cast<size_t>(output_width) * output_height;
    }
    if (plane_elements > SIZE_MAX / sizeof(long long) / MATCH_PLANE_COUNT) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaSuccess;
    int result = reserve_device(
        &persistent->u8[MATCH_TEMPLATE_BUFFER], &persistent->u8_capacity[MATCH_TEMPLATE_BUFFER],
        template_bytes, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // The gray ROI is fully rewritten by this call, so it must be exactly the requested size:
    // a larger leftover buffer from an earlier call would keep the old row pitch.
    result = reserve_exact(
        &persistent->u8[MATCH_ROI_BUFFER], &persistent->u8_capacity[MATCH_ROI_BUFFER],
        static_cast<size_t>(search_width) * search_height, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // planes: sum prefix, square prefix
    for (int plane = 0; plane < MATCH_PLANE_COUNT; ++plane) {
        result = reserve_device(
            &persistent->match_plane[plane], &persistent->match_plane_capacity[plane],
            plane_elements, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
    }
    result = reserve_device(
        &persistent->match_candidates, &persistent->match_candidate_capacity,
        static_cast<size_t>(match_slot_count(output_width)), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    persistent->match_candidate_output_width = output_width;
    long long* candidate_storage = persistent->match_candidates;
    unsigned long long* best_keys = reinterpret_cast<unsigned long long*>(candidate_storage);
    double* candidate_scores =
        reinterpret_cast<double*>(candidate_storage + match_score_offset(output_width));
    int* candidate_ys = reinterpret_cast<int*>(candidate_storage + match_row_offset(output_width));
    // Result slots live in the same grow-only block: two int32 for the winner origin, one float
    // score and one packed best key, all int64-aligned.
    const int result_offset = match_result_offset(output_width);
    const int key_offset = match_result_offset(output_width) + 1;
    int* match_xy_device = reinterpret_cast<int*>(candidate_storage + result_offset);
    float* match_score_device = reinterpret_cast<float*>(match_xy_device + 2);
    unsigned long long* best_key = reinterpret_cast<unsigned long long*>(candidate_storage + key_offset);

    // Template statistics on the host: the template is small and constant per Recipe.
    double template_sum = 0.0;
    double template_square_sum = 0.0;
    for (int row = 0; row < template_height; ++row) {
        for (int column = 0; column < template_width; ++column) {
            const int value = templ[static_cast<size_t>(row) * template_width + column];
            template_sum += value;
            template_square_sum += static_cast<double>(value) * value;
        }
    }
    const double template_pixels = static_cast<double>(template_width) * template_height;
    const double template_mean = template_sum / template_pixels;
    const double template_variance = template_square_sum / template_pixels - template_mean * template_mean;
    if (!(template_variance > 1e-12)) {
        // core/tiler.py::_find_grid_anchor switches to TM_SQDIFF_NORMED when the template has no
        // contrast. The extension's difference path is not equivalent to OpenCV yet (it reports
        // scores outside [0, 1] and can pick a neighbouring column), so the caller restarts this
        // step on the CPU reference instead of receiving a wrong anchor. Production anchor
        // templates are structured patterns, so this affects the flat-template edge case only.
        return VF_CUDA_UNSUPPORTED;
    }
    if (output_width > MATCH_COORD_MASK) return VF_CUDA_INVALID_ARGUMENT;
    reset_timing(persistent, true);
    error = cudaMemcpyAsync(
        persistent->u8[MATCH_TEMPLATE_BUFFER], templ, template_bytes,
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);

    // Gray the search ROI on the device: the resident image is BGR, the reference works on gray,
    // and the weights are the ones vf_bgr_to_gray_u8 already uses. The ROI-aware kernel is needed
    // because a resident sub-rectangle is not tightly packed.
    const size_t resident_pitch =
        static_cast<size_t>(persistent->resident_width) * persistent->resident_channels;
    const uint8_t* roi_source = persistent->resident_u8 +
        static_cast<size_t>(search_y) * resident_pitch +
        static_cast<size_t>(search_x) * persistent->resident_channels;
    if (persistent->resident_channels == 3) {
        bgr_gray_roi_kernel<<<
            grid2d(search_width, search_height), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            roi_source, persistent->resident_width, 0, 0,
            persistent->u8[MATCH_ROI_BUFFER], search_width, search_height);
    } else {
        error = cudaMemcpy2DAsync(
            persistent->u8[MATCH_ROI_BUFFER], static_cast<size_t>(search_width), roi_source,
            resident_pitch, static_cast<size_t>(search_width), search_height,
            cudaMemcpyDeviceToDevice, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
    }
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    // Shrink the tile until its shared footprint fits, because a large template widens the halo.
    int tile_cols = MATCH_TILE_COLS;
    int tile_rows = MATCH_TILE_ROWS;
    size_t shared_bytes = 0;
    const size_t halo = static_cast<size_t>(template_height - 1);
    for (;;) {
        shared_bytes = (static_cast<size_t>(tile_rows) + halo) *
                       (static_cast<size_t>(tile_cols) + static_cast<size_t>(template_width - 1));
        if (shared_bytes <= MATCH_SHARED_LIMIT_BYTES) break;
        // Shrink the larger dimension first, and never below a one-row-tall, 8-column-wide tile,
        // which keeps the block occupied while reducing the halo overhead.
        if (tile_cols >= tile_rows && tile_cols > 8) {
            tile_cols = tile_cols > 16 ? tile_cols / 2 : tile_cols - 4;
            continue;
        }
        if (tile_rows > 1) {
            tile_rows = tile_rows > 2 ? tile_rows / 2 : tile_rows - 1;
            continue;
        }
        break;
    }
    // When the ROI tile had to shrink below the tile we would like, the template-in-shared layout
    // is the better trade: the template is the small operand and the ROI streams through L2.
    const bool use_shared_template =
        (tile_cols < MATCH_TILE_COLS || tile_rows < MATCH_TILE_ROWS) &&
        template_bytes <= MATCH_SHARED_LIMIT_BYTES;
    if (shared_bytes > MATCH_SHARED_LIMIT_BYTES && !use_shared_template) {
        // Beyond this the template height alone exceeds the block budget, so the caller restarts
        // localization on the CPU reference instead of receiving a wrong anchor.
        return VF_CUDA_UNSUPPORTED;
    }
    if (cudaFuncSetAttribute(
            match_score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(MATCH_SHARED_LIMIT_BYTES)) != cudaSuccess) {
        return cuda_result(cudaGetLastError());
    }
    if (use_shared_template) {
        if (cudaFuncSetAttribute(
                match_score_shared_template_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(template_bytes)) != cudaSuccess) {
            return cuda_result(cudaGetLastError());
        }
        const dim3 shared_grid(
            static_cast<unsigned int>((output_width + MATCH_BLOCK_X - 1) / MATCH_BLOCK_X),
            static_cast<unsigned int>((output_height + MATCH_BLOCK_Y - 1) / MATCH_BLOCK_Y));
        const dim3 shared_block(MATCH_BLOCK_X, MATCH_BLOCK_Y);
        error = cudaMemsetAsync(
            best_keys, 0, sizeof(unsigned long long) * match_candidate_slots(output_width),
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        match_score_shared_template_kernel<<<
            shared_grid, shared_block, template_bytes, persistent->stream>>>(
            persistent->u8[MATCH_ROI_BUFFER], search_width,
            output_width, output_height, template_width, template_height,
            template_width * template_height, MATCH_BLOCK_X,
            template_mean, template_variance,
            persistent->u8[MATCH_TEMPLATE_BUFFER], best_keys);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    } else {
    const int tiles_x = (output_width + tile_cols - 1) / tile_cols;
    const int tiles_y = (output_height + tile_rows - 1) / tile_rows;
    const dim3 score_grid(static_cast<unsigned int>(tiles_x) * static_cast<unsigned int>(tiles_y));
    const dim3 score_block(MATCH_BLOCK_X, MATCH_BLOCK_Y);
        // The per-column key slots are the compare-and-swap targets, so they must start empty.
        error = cudaMemsetAsync(
            best_keys, 0, sizeof(unsigned long long) * match_candidate_slots(output_width),
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        match_score_kernel<<<score_grid, score_block, shared_bytes, persistent->stream>>>(
            persistent->u8[MATCH_ROI_BUFFER], search_width,
            output_width, output_height, template_width, template_height,
            template_width * template_height, tile_cols, tile_rows,
            template_mean, template_variance,
            persistent->u8[MATCH_TEMPLATE_BUFFER],
            best_keys);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
    }

    // Reduce the per-column keys to one global best with the same packed ordering, then unpack it
    // on the device so the host receives coordinates and the quantized score.
    error = cudaMemsetAsync(best_key, 0, sizeof(unsigned long long), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    match_publish_kernel<<<
        dim3((output_width + MATCH_REDUCE_BLOCK - 1) / MATCH_REDUCE_BLOCK, 1),
        dim3(MATCH_REDUCE_BLOCK, 1), 0, persistent->stream>>>(
        best_keys, output_width, best_key);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    (void)candidate_scores;
    (void)candidate_ys;

    match_unpack_kernel<<<1, 1, 0, persistent->stream>>>(
        best_key, 1.0f, match_xy_device, match_score_device);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    int match_xy[2] = {0, 0};
    error = cudaMemcpyAsync(
        match_xy, match_xy_device, sizeof(match_xy), cudaMemcpyDeviceToHost, persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            out_score, match_score_device, sizeof(float), cudaMemcpyDeviceToHost, persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result == VF_CUDA_OK) finalize_timing(persistent);
    if (result != VF_CUDA_OK) return result;
    if (match_xy[0] < 0 || match_xy[1] < 0) return VF_CUDA_INTERNAL_ERROR;

    out_match[0] = search_x + match_xy[0];
    out_match[1] = search_y + match_xy[1];
    out_match[2] = template_width;
    out_match[3] = template_height;
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_match_template_debug_key(void* context, unsigned long long* out_key) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_key == nullptr || persistent->match_candidates == nullptr) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int key_offset =
        match_result_offset(static_cast<int>(persistent->match_candidate_output_width)) + 1;
    const unsigned long long* best_key = reinterpret_cast<const unsigned long long*>(
        persistent->match_candidates + key_offset);
    cudaError_t error = cudaMemcpyAsync(
        out_key, best_key, sizeof(unsigned long long), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy one prefix plane back for comparison against the CPU reference.
VF_CUDA_API int vf_match_template_debug_planes(
    void* context, int plane, int64_t* out_values, size_t count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_values == nullptr || count == 0 ||
        plane < 0 || plane >= MATCH_PLANE_COUNT || persistent->match_plane[plane] == nullptr ||
        count > persistent->match_plane_capacity[plane]) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_values, persistent->match_plane[plane], count * sizeof(int64_t),
        cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy the gray search ROI back so a caller can compare it with the CPU gray image.
VF_CUDA_API int vf_match_template_debug_roi(void* context, uint8_t* out_values, size_t count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_values == nullptr || count == 0 ||
        persistent->u8[MATCH_ROI_BUFFER] == nullptr ||
        count > persistent->u8_capacity[MATCH_ROI_BUFFER]) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_values, persistent->u8[MATCH_ROI_BUFFER], count, cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Diagnostics: copy the per-column candidate scores and rows of the last localization call, so a
// caller can compare each column's best against the CPU match map. candidate_slots reports how
// many slots the context currently holds.
VF_CUDA_API int vf_match_template_debug_candidates(
    void* context, double* out_scores, int* out_rows, int count, int* candidate_slots) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (candidate_slots != nullptr) {
        *candidate_slots = static_cast<int>(persistent != nullptr
            ? persistent->match_candidate_capacity : 0);
    }
    if (persistent == nullptr || out_scores == nullptr || out_rows == nullptr || count <= 0 ||
        persistent->match_candidates == nullptr || count > MATCH_MAX_OUTPUT_WIDTH ||
        count > persistent->match_candidate_output_width ||
        count * MATCH_CANDIDATE_SLOT_STRIDE > (int)persistent->match_candidate_capacity) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const double* scores = reinterpret_cast<const double*>(persistent->match_candidates);
    const int* rows = reinterpret_cast<const int*>(
        persistent->match_candidates + match_candidate_slots(count));
    cudaError_t error = cudaMemcpyAsync(
        out_scores, scores, sizeof(double) * count, cudaMemcpyDeviceToHost, persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            out_rows, rows, sizeof(int) * count, cudaMemcpyDeviceToHost, persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Contour trace of the requested region of the resident binary mask. Scratch buffers are grow-only;
// when the first guess at the output size is too small the kernel reports the exact requirement and
// the whole trace is re-run with that capacity, so a result is never truncated silently.
VF_CUDA_API int vf_find_contours_u8(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height, int mode,
    int* out_contour_count, int* out_point_count) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_contour_count == nullptr || out_point_count == nullptr ||
        generation == 0 || generation != persistent->resident_generation ||
        persistent->resident_u8 == nullptr ||
        (mode != VF_CONTOURS_RETR_EXTERNAL && mode != VF_CONTOURS_RETR_LIST)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    // A colour resident image cannot be reinterpreted as a binary mask without changing the
    // foreground rule, so the caller restarts this step on the CPU reference instead.
    if (persistent->resident_channels != 1) return VF_CUDA_UNSUPPORTED;
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x > persistent->resident_width - width ||
        y > persistent->resident_height - height) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int label_stride = width + 2;
    const size_t padded = static_cast<size_t>(label_stride) * static_cast<size_t>(height + 2);
    if (padded > static_cast<size_t>(INT_MAX)) return VF_CUDA_INVALID_ARGUMENT;

    const size_t resident_pitch =
        static_cast<size_t>(persistent->resident_width) * persistent->resident_channels;
    // The init kernel applies (x, y) itself: passing an already-offset pointer here would apply
    // the region origin twice and silently trace the wrong window.
    const uint8_t* mask = persistent->resident_u8;

    // First guess at the output size. Sparse production masks are far below these ratios; a denser
    // mask only costs one extra trace with the exact reported capacity.
    int contour_hint = static_cast<int>(std::min<long long>(
        std::max<long long>(static_cast<long long>(padded) / 64, 64), 1LL << 20));
    int point_hint = static_cast<int>(std::min<long long>(
        std::max<long long>(static_cast<long long>(padded) / 8, 256), 1LL << 22));

    int counts[3] = {0, 0, 0};
    int result = reserve_device(
        &persistent->contour_label, &persistent->contour_label_capacity, padded,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    // The scan reports the contour count, the point count and the overflow flag in one block.
    result = reserve_device(
        &persistent->contour_counts, &persistent->contour_count_capacity, static_cast<size_t>(4),
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, false);
    cudaError_t error = cudaSuccess;
    // RETR_LIST walks the zero-ness transition list instead of every row byte. The list depends
    // only on the mask, so it is built once, before the retry loop, and its size is exact (the
    // per-row counts come back to the host and are prefix-summed there).
    const bool list_mode = (mode == VF_CONTOURS_RETR_LIST);
    if (list_mode) {
        std::vector<int> row_counts;
        std::vector<int> row_start;
        try {
            row_counts.assign(static_cast<size_t>(height), 0);
            row_start.assign(static_cast<size_t>(height) + 2, 0);
        } catch (const std::bad_alloc&) {
            return VF_CUDA_ALLOCATION_FAILED;
        }
        result = reserve_device(
            &persistent->contour_row_counts, &persistent->contour_row_count_capacity,
            static_cast<size_t>(height), &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        contour_row_transition_counts_kernel<<<
            dim3(static_cast<unsigned int>((height + 255) / 256), 1, 1), dim3(256, 1, 1), 0,
            persistent->stream>>>(
            mask, static_cast<int>(resident_pitch), x, y, width, height,
            persistent->contour_row_counts);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        error = cudaMemcpyAsync(
            row_counts.data(), persistent->contour_row_counts,
            sizeof(int) * static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;

        long long total = 0;
        for (int row = 1; row <= height; ++row) {
            row_start[row] = static_cast<int>(total);
            total += row_counts[static_cast<size_t>(row - 1)];
        }
        if (total > INT_MAX) return VF_CUDA_INVALID_ARGUMENT;
        row_start[height + 1] = static_cast<int>(total);

        result = reserve_device(
            &persistent->contour_transitions, &persistent->contour_transition_capacity,
            static_cast<size_t>(total > 0 ? total : 1), &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_row_start, &persistent->contour_row_start_capacity,
            static_cast<size_t>(height) + 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        error = cudaMemcpyAsync(
            persistent->contour_row_start, row_start.data(),
            sizeof(int) * (static_cast<size_t>(height) + 2), cudaMemcpyHostToDevice,
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        if (total > 0) {
            contour_fill_transitions_kernel<<<
                dim3(static_cast<unsigned int>((height + 255) / 256), 1, 1), dim3(256, 1, 1), 0,
                persistent->stream>>>(
                mask, static_cast<int>(resident_pitch), x, y, width, height,
                persistent->contour_row_start, label_stride, persistent->contour_transitions);
            result = visionflow_cuda::kernel_launch_result();
            if (result != VF_CUDA_OK) return result;
        }
    }

    bool complete = false;
    for (int attempt = 0; attempt < 4 && !complete; ++attempt) {
        result = reserve_device(
            &persistent->contour_offsets, &persistent->contour_offset_capacity,
            static_cast<size_t>(contour_hint) + 1, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_points, &persistent->contour_point_capacity,
            static_cast<size_t>(point_hint) * 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_out_offsets, &persistent->contour_out_offset_capacity,
            static_cast<size_t>(contour_hint) + 1, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;
        result = reserve_device(
            &persistent->contour_out_points, &persistent->contour_out_point_capacity,
            static_cast<size_t>(point_hint) * 2, &persistent->allocation_count);
        if (result != VF_CUDA_OK) return result;

        contour_init_label_kernel<<<
            grid2d(label_stride, height + 2), dim3(BLOCK_X, BLOCK_Y), 0, persistent->stream>>>(
            mask, static_cast<int>(resident_pitch), x, y, width, height,
            persistent->contour_label, label_stride);
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        cudaEventRecord(persistent->timing_events[TIMING_AFTER_INPUT], persistent->stream);

        if (list_mode) {
            contour_scan_list_kernel<<<1, 1, 0, persistent->stream>>>(
                persistent->contour_label, label_stride, height,
                persistent->contour_transitions, persistent->contour_row_start,
                persistent->contour_offsets, static_cast<int>(persistent->contour_offset_capacity),
                persistent->contour_points, static_cast<int>(persistent->contour_point_capacity / 2),
                persistent->contour_counts);
        } else {
            contour_scan_kernel<<<1, 1, 0, persistent->stream>>>(
                persistent->contour_label, label_stride, width, height, mode,
                persistent->contour_offsets, static_cast<int>(persistent->contour_offset_capacity),
                persistent->contour_points, static_cast<int>(persistent->contour_point_capacity / 2),
                persistent->contour_counts);
        }
        result = visionflow_cuda::kernel_launch_result();
        if (result != VF_CUDA_OK) return result;
        cudaEventRecord(persistent->timing_events[TIMING_AFTER_KERNEL], persistent->stream);

        error = cudaMemcpyAsync(
            counts, persistent->contour_counts, sizeof(counts), cudaMemcpyDeviceToHost,
            persistent->stream);
        if (error != cudaSuccess) return cuda_result(error);
        cudaEventRecord(persistent->timing_events[TIMING_AFTER_OUTPUT], persistent->stream);
        result = visionflow_cuda::stream_result(persistent->stream);
        if (result != VF_CUDA_OK) return result;

        if (counts[2] != 0) {
            // Grow to the exact capacity the trace reported and run it again; the second run has
            // enough room by construction, and identical input always yields identical counts.
            contour_hint = counts[0];
            point_hint = counts[1];
            continue;
        }

        if (counts[0] > 0) {
            contour_reverse_kernel<<<
                dim3(static_cast<unsigned int>((counts[0] + 127) / 128), 1, 1), dim3(128, 1, 1),
                0, persistent->stream>>>(
                persistent->contour_offsets, persistent->contour_points, counts[0], counts[1],
                persistent->contour_out_offsets, persistent->contour_out_points);
            result = visionflow_cuda::kernel_launch_result();
            if (result != VF_CUDA_OK) return result;
        }
        complete = true;
    }
    if (!complete) return VF_CUDA_INTERNAL_ERROR;
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);

    persistent->contour_count = counts[0];
    persistent->contour_point_count = counts[1];
    persistent->contour_generation = generation;
    persistent->contour_result_valid = true;
    *out_contour_count = counts[0];
    *out_point_count = counts[1];
    return VF_CUDA_OK;
}

// Copies the most recent contour result to the caller. A short buffer is an error, never a partial
// result, and a stale result (the resident image changed after the trace) is rejected.
VF_CUDA_API int vf_find_contours_download(
    void* context,
    int32_t* out_offsets, int offset_capacity,
    int32_t* out_points, int point_capacity) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || out_offsets == nullptr || !persistent->contour_result_valid ||
        persistent->contour_generation != persistent->resident_generation) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int contour_count = persistent->contour_count;
    const int point_count = persistent->contour_point_count;
    if (offset_capacity < contour_count + 1) return VF_CUDA_INVALID_ARGUMENT;
    if (point_count > 0 && (out_points == nullptr || point_capacity < point_count)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (contour_count == 0) {
        out_offsets[0] = 0;
        return VF_CUDA_OK;
    }
    cudaError_t error = cudaMemcpyAsync(
        out_offsets, persistent->contour_out_offsets,
        sizeof(int32_t) * static_cast<size_t>(contour_count + 1), cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error == cudaSuccess && point_count > 0) {
        error = cudaMemcpyAsync(
            out_points, persistent->contour_out_points,
            sizeof(int32_t) * 2 * static_cast<size_t>(point_count), cudaMemcpyDeviceToHost,
            persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    return visionflow_cuda::stream_result(persistent->stream);
}

// Monotone float32 -> uint32 order key: positives keep their magnitude and gain the top bit,
// negatives invert every bit. Unsigned integer order then equals float order for every bit pattern,
// including -0.0 < +0.0, subnormals and infinities, so a plain key sort orders floats exactly.
__global__ void median_order_key_kernel(
    const float* values,
    uint32_t* keys,
    int* nan_flag,
    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint32_t bits = __float_as_uint(values[index]);
    keys[index] = (bits & 0x80000000u) != 0u
        ? (bits ^ 0xFFFFFFFFu)
        : (bits | 0x80000000u);
    // NumPy's median returns NaN whenever the input holds one (_median_nancheck inspects the last
    // partitioned element, and every NaN sorts last), so NaN presence is reported explicitly rather
    // than being folded into an invented key order.
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        atomicOr(nan_flag, 1);
    }
}

// Inverse of median_order_key_kernel() for the one or two middle keys, evaluated on the host.
float median_key_to_value(uint32_t key) {
    const uint32_t bits = (key & 0x80000000u) != 0u
        ? (key & 0x7FFFFFFFu)
        : (key ^ 0xFFFFFFFFu);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Bit-exact np.median for a host float32 array. The full contract is documented in
// include/visionflow_cuda.h: monotone keys, device radix sort, middle-key-only readback, and the
// float32 even-count average on the host.
VF_CUDA_API int vf_median_f32(
    void* context,
    const float* values,
    long long count,
    float* out_median) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || values == nullptr || out_median == nullptr || count <= 0 ||
        count > static_cast<long long>(INT_MAX)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int items = static_cast<int>(count);
    const size_t item_count = static_cast<size_t>(items);

    int result = reserve_device(
        &persistent->median_values, &persistent->median_value_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->median_keys, &persistent->median_key_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->median_sorted_keys, &persistent->median_sorted_key_capacity, item_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->median_nan_flag, &persistent->median_nan_flag_capacity,
        static_cast<size_t>(1), &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    // Size the CUB temporary storage with the same offset type the sorting call below uses.
    size_t sort_storage_bytes = 0;
    cudaError_t error = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_storage_bytes, persistent->median_keys, persistent->median_sorted_keys,
        items, 0, static_cast<int>(sizeof(uint32_t) * 8), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    const size_t sort_storage_required = sort_storage_bytes;
    if (sort_storage_required == 0) return VF_CUDA_INTERNAL_ERROR;
    result = reserve_device(
        &persistent->median_sort_scratch, &persistent->median_sort_scratch_capacity,
        sort_storage_required, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, true);
    error = cudaMemcpyAsync(
        persistent->median_values, values, sizeof(float) * item_count, cudaMemcpyHostToDevice,
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_INPUT], persistent->stream);

    error = cudaMemsetAsync(persistent->median_nan_flag, 0, sizeof(int), persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    constexpr int MEDIAN_THREADS = 256;
    median_order_key_kernel<<<
        (items + MEDIAN_THREADS - 1) / MEDIAN_THREADS, MEDIAN_THREADS, 0, persistent->stream>>>(
        persistent->median_values, persistent->median_keys, persistent->median_nan_flag, items);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;

    error = cub::DeviceRadixSort::SortKeys(
        persistent->median_sort_scratch, sort_storage_bytes, persistent->median_keys,
        persistent->median_sorted_keys, items, 0, static_cast<int>(sizeof(uint32_t) * 8),
        persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_KERNEL], persistent->stream);

    // Only the middle one or two keys cross PCIe, plus the four-byte NaN-presence word.
    const bool even = (items % 2) == 0;
    const int middle = items / 2;
    const int first_key = even ? (middle - 1) : middle;
    const int key_reads = even ? 2 : 1;
    uint32_t host_keys[2] = {0u, 0u};
    int host_nan = 0;
    error = cudaMemcpyAsync(
        host_keys, persistent->median_sorted_keys + first_key,
        sizeof(uint32_t) * static_cast<size_t>(key_reads), cudaMemcpyDeviceToHost,
        persistent->stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            &host_nan, persistent->median_nan_flag, sizeof(int), cudaMemcpyDeviceToHost,
            persistent->stream);
    }
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_OUTPUT], persistent->stream);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);

    if (host_nan != 0) {
        const uint32_t quiet_nan_bits = 0x7FC00000u;
        std::memcpy(out_median, &quiet_nan_bits, sizeof(*out_median));
        return VF_CUDA_OK;
    }
    const float low = median_key_to_value(host_keys[0]);
    if (!even) {
        *out_median = low;
        return VF_CUDA_OK;
    }
    const float high = median_key_to_value(host_keys[1]);
    // np.median averages the two middle values in the input dtype: a float32 add, then a float32
    // divide by two. Reproduce both operations exactly, including their overflow and rounding.
    *out_median = (low + high) / 2.0f;
    return VF_CUDA_OK;
}

// Shared body of vf_gaussian_blur_f32 and vf_gaussian_blur_f32_roi. The requested rectangle of the
// host float32 source is uploaded with a 2D copy (only the rectangle - the ROI export never
// touches pixels outside it, which is what makes it equal to cv2.GaussianBlur on the same
// sub-array), blurred on the device with the context's grow-only scratch, and copied back.
static int gaussian_blur_f32_device(
    PersistentContext* persistent,
    const float* src, int src_stride, int offset_x, int offset_y,
    int width, int height, float* dst, int dst_stride, int kernel_size) {
    const size_t row_values = static_cast<size_t>(width);
    const size_t pixel_count = row_values * static_cast<size_t>(height);
    if (pixel_count == 0 || pixel_count > SIZE_MAX / sizeof(float)) return VF_CUDA_INVALID_ARGUMENT;
    const size_t row_bytes = row_values * sizeof(float);
    int result = reserve_device(
        &persistent->gaussian_f32_input, &persistent->gaussian_f32_input_capacity, pixel_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->gaussian_f32_intermediate, &persistent->gaussian_f32_intermediate_capacity,
        pixel_count, &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;
    result = reserve_device(
        &persistent->gaussian_f32_output, &persistent->gaussian_f32_output_capacity, pixel_count,
        &persistent->allocation_count);
    if (result != VF_CUDA_OK) return result;

    reset_timing(persistent, true);
    cudaError_t error = cudaMemcpy2DAsync(
        persistent->gaussian_f32_input, row_bytes,
        reinterpret_cast<const char*>(src) + static_cast<size_t>(offset_y) * src_stride +
            static_cast<size_t>(offset_x) * sizeof(float),
        static_cast<size_t>(src_stride), row_bytes, static_cast<size_t>(height),
        cudaMemcpyHostToDevice, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_INPUT], persistent->stream);

    int radius = 0;
    result = prepare_gaussian_f32_weights(kernel_size, &radius, persistent->stream);
    if (result != VF_CUDA_OK) return result;
    launch_gaussian_f32(
        persistent->gaussian_f32_input, persistent->gaussian_f32_intermediate,
        persistent->gaussian_f32_output, width, height, radius, persistent->stream);
    result = visionflow_cuda::kernel_launch_result();
    if (result != VF_CUDA_OK) return result;
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_KERNEL], persistent->stream);

    error = cudaMemcpy2DAsync(
        dst, static_cast<size_t>(dst_stride), persistent->gaussian_f32_output, row_bytes,
        row_bytes, static_cast<size_t>(height), cudaMemcpyDeviceToHost, persistent->stream);
    if (error != cudaSuccess) return cuda_result(error);
    cudaEventRecord(persistent->timing_events[TIMING_AFTER_OUTPUT], persistent->stream);
    result = visionflow_cuda::stream_result(persistent->stream);
    if (result != VF_CUDA_OK) return result;
    finalize_timing(persistent);
    return VF_CUDA_OK;
}

// Kernel sizes below 3 are a malformed request; odd sizes outside the verified range are reported
// as unsupported rather than computed unvalidated.
static int gaussian_f32_kernel_request(int kernel_size) {
    if (kernel_size < GAUSSIAN_F32_MIN_KERNEL) return VF_CUDA_INVALID_ARGUMENT;
    return gaussian_f32_kernel_supported(kernel_size) ? VF_CUDA_OK : VF_CUDA_UNSUPPORTED;
}

// Separable float32 Gaussian with reflect101 borders, reproducing
// cv2.GaussianBlur(single_channel_float32, (ksize, ksize), 0.0) within the tolerance documented in
// include/visionflow_cuda.h. Strides are byte counts, as everywhere else in this ABI.
VF_CUDA_API int vf_gaussian_blur_f32(
    void* context,
    const float* src, int width, int height, int src_stride,
    float* dst, int dst_stride,
    int kernel_size) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || src == nullptr || dst == nullptr || width <= 0 || height <= 0 ||
        width > INT_MAX / static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int minimum_stride = width * static_cast<int>(sizeof(float));
    if (src_stride < minimum_stride || dst_stride < minimum_stride) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    return gaussian_blur_f32_device(
        persistent, src, src_stride, 0, 0, width, height, dst, dst_stride, kernel_size);
}

// Same operator restricted to one rectangle of a wider host float32 source. The rectangle is
// treated as an isolated image - borders reflect inside it, pixels outside it are never read - so
// the result equals cv2.GaussianBlur(src[y:y+height, x:x+width], (ksize, ksize), 0.0) and only the
// rectangle crosses PCIe.
VF_CUDA_API int vf_gaussian_blur_f32_roi(
    void* context,
    const float* src, int src_width, int src_height, int src_stride,
    int x, int y, int width, int height,
    float* dst, int dst_stride,
    int kernel_size) {
    PersistentContext* persistent = static_cast<PersistentContext*>(context);
    if (persistent == nullptr || src == nullptr || dst == nullptr ||
        src_width <= 0 || src_height <= 0 || width <= 0 || height <= 0 ||
        x < 0 || y < 0 || width > src_width || height > src_height ||
        x > src_width - width || y > src_height - height ||
        src_width > INT_MAX / static_cast<int>(sizeof(float)) ||
        width > INT_MAX / static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    if (src_stride < src_width * static_cast<int>(sizeof(float)) ||
        dst_stride < width * static_cast<int>(sizeof(float))) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    const int kernel_status = gaussian_f32_kernel_request(kernel_size);
    if (kernel_status != VF_CUDA_OK) return kernel_status;
    return gaussian_blur_f32_device(
        persistent, src, src_stride, x, y, width, height, dst, dst_stride, kernel_size);
}
