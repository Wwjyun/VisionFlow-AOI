#ifndef VISIONFLOW_CUDA_H
#define VISIONFLOW_CUDA_H

#include <stdint.h>
#include "visionflow_cuda_errors.h"

#define VF_CUDA_ABI_VERSION 1
#define VF_CUDA_PLAN_VERSION 1
#define VF_PLAN_INPUT_NODE (-1)

/*
 * ABI rules:
 * - All image pointers are host pointers to uint8 interleaved data.
 * - Strides are byte counts, not pixel counts.
 * - The caller owns every input/output buffer and must allocate the output.
 * - Calls are synchronous: output is ready when the function returns.
 * - A return value of VF_CUDA_OK means success; other values are declared in
 *   visionflow_cuda_errors.h and can be described by vf_gpu_error_message().
 * - The Python bridge serializes calls sharing one GpuRuntime. Native callers
 *   should also serialize calls unless they provide their own higher-level
 *   synchronization.
 * - Context APIs are additive ABI v1 extensions. Callers may probe their
 *   exports and keep using the stateless primitive APIs with an older DLL.
 * - A context owns reusable device buffers and must be destroyed by the same
 *   module with vf_context_destroy(). It is not safe for concurrent calls.
 */

#if defined(_WIN32)
#  if defined(VISIONFLOW_CUDA_EXPORTS)
#    define VF_CUDA_API __declspec(dllexport)
#  else
#    define VF_CUDA_API __declspec(dllimport)
#  endif
#else
#  define VF_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum VisionFlowMorphologyOperation {
    VF_MORPH_OPEN = 0,
    VF_MORPH_CLOSE = 1,
    VF_MORPH_DILATE = 2,
    VF_MORPH_ERODE = 3
};

enum VisionFlowPlanOperatorKind {
    VF_PLAN_GRAY = 1,
    VF_PLAN_GAUSSIAN = 2,
    VF_PLAN_THRESHOLD = 3,
    VF_PLAN_ADAPTIVE_MEAN = 4,
    VF_PLAN_MORPHOLOGY = 5,
    VF_PLAN_RESIZE_AREA = 6
};

typedef struct VfPlanOperatorV1 {
    uint32_t struct_size;
    int32_t kind;
    int32_t input_node;
    int32_t output_node;
    int32_t int_params[4];
    float float_params[2];
} VfPlanOperatorV1;

typedef struct VfPlanDescV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t input_channels;
    int32_t operator_count;
    const VfPlanOperatorV1* operators;
    int32_t output_node;
} VfPlanDescV1;

typedef struct VfDagPlanDescV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t input_channels;
    int32_t operator_count;
    const VfPlanOperatorV1* operators;
    int32_t output_count;
    const int32_t* output_nodes;
} VfDagPlanDescV1;

typedef struct VfDagOutputV1 {
    uint32_t struct_size;
    int32_t node;
    uint8_t* data;
    int32_t stride;
    int32_t channels;
} VfDagOutputV1;

typedef struct VfRoiV1 {
    uint32_t struct_size;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
} VfRoiV1;

typedef struct VfCudaTimingsV1 {
    uint32_t struct_size;
    uint32_t version;
    float context_create_ms;
    float allocation_ms;
    float h2d_ms;
    float device_copy_ms;
    float kernel_ms;
    float d2h_ms;
    float synchronize_ms;
    float free_ms;
    float gaussian_ms;
    float adaptive_integral_ms;
    float threshold_ms;
    float morphology_ms;
    float total_device_ms;
} VfCudaTimingsV1;

VF_CUDA_API int vf_gpu_abi_version(void);
VF_CUDA_API int vf_gpu_device_count(void);
VF_CUDA_API int vf_gpu_compute_capability(void);
VF_CUDA_API int vf_gpu_device_name(char* output, int capacity);
VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity);
VF_CUDA_API int vf_gpu_memory_info(uint64_t* free_bytes, uint64_t* total_bytes);

VF_CUDA_API int vf_context_create(void** context);
VF_CUDA_API int vf_context_destroy(void* context);
VF_CUDA_API int vf_context_stats(
    void* context, uint64_t* reserved_bytes, uint64_t* allocation_count);
VF_CUDA_API int vf_context_last_timings(void* context, VfCudaTimingsV1* timings);
VF_CUDA_API int vf_context_upload_u8(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint64_t* generation);
VF_CUDA_API int vf_roi_batch_create(
    void* context, uint64_t generation,
    const VfRoiV1* rois, int roi_count, void** batch);
VF_CUDA_API int vf_roi_batch_info(
    void* batch, int* roi_count, int* width, int* height, int* channels);
VF_CUDA_API int vf_roi_batch_download_u8(
    void* batch, int roi_index,
    uint8_t* dst, int dst_stride, int dst_channels);
VF_CUDA_API int vf_roi_batch_destroy(void* batch);

/*
 * Optional generic plan ABI. The descriptor is backend-neutral and contains
 * no detector ID/name. vf_plan_create copies and validates the descriptor;
 * vf_plan_execute only transfers image data and launches the compiled plan.
 * A plan borrows its context and must be destroyed before that context.
 */
VF_CUDA_API int vf_plan_query(
    const VfPlanDescV1* desc, int width, int height,
    char* reason, int reason_capacity);
VF_CUDA_API int vf_plan_create(
    void* context, const VfPlanDescV1* desc, int width, int height, void** plan);
VF_CUDA_API int vf_plan_execute(
    void* plan,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);
VF_CUDA_API int vf_plan_destroy(void* plan);
VF_CUDA_API int vf_plan_execute_roi(
    void* plan, uint64_t generation, int x, int y,
    uint8_t* dst, int dst_stride, int dst_channels);

/*
 * Optional detector-neutral DAG extension. Nodes are topologically ordered,
 * may reference the root or an earlier node, and may expose multiple named-by-
 * index outputs. Execution uploads the root once and synchronizes after all
 * requested host outputs have been copied.
 */
VF_CUDA_API int vf_dag_plan_query(
    const VfDagPlanDescV1* desc, int width, int height,
    char* reason, int reason_capacity);
VF_CUDA_API int vf_dag_plan_create(
    void* context, const VfDagPlanDescV1* desc, int width, int height, void** plan);
VF_CUDA_API int vf_dag_plan_execute(
    void* plan,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    const VfDagOutputV1* outputs, int output_count);
VF_CUDA_API int vf_dag_plan_destroy(void* plan);
VF_CUDA_API int vf_dag_plan_execute_roi(
    void* plan, uint64_t generation, int x, int y,
    const VfDagOutputV1* outputs, int output_count);

VF_CUDA_API int vf_bgr_to_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_bgr_to_rgb_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_crop_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int crop_x, int crop_y, int crop_width, int crop_height);

VF_CUDA_API int vf_resize_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int dst_width, int dst_height);

VF_CUDA_API int vf_gaussian_blur_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int kernel_size);

VF_CUDA_API int vf_threshold_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int threshold, int max_value, int invert);

VF_CUDA_API int vf_adaptive_mean_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int block_size, float c, int max_value, int invert);

VF_CUDA_API int vf_morphology_rect_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int operation, int kernel_size, int iterations);

/*
 * Optional Template Anchor Grid localization extension. Reads a search ROI out of the
 * context's resident image, converts it to gray with the same weights as
 * vf_bgr_to_gray_u8, and returns the best TM_CCOEFF_NORMED match of a host gray template as
 * a half-open rectangle plus its score. Only the result rectangle and score cross PCIe; the
 * template is uploaded once per call (it is small and constant per Recipe).
 *
 * Semantics match the CPU reference in core/tiler.py::Tiler._find_grid_anchor:
 *   score = sum((window - mean(window)) * (templ - mean(templ)))
 *           / sqrt(sum((window - mean(window))^2) * sum((templ - mean(templ))^2))
 * Ties (equal score) resolve to the topmost, then leftmost match. The template must not be
 * flat (standard deviation <= 1e-6); the CPU reference switches to TM_SQDIFF_NORMED there,
 * which is reported as VF_CUDA_UNSUPPORTED so the caller can restart the step on the CPU.
 * out_match must hold 4 int32 values [x, y, width, height] and out_score 1 float.
 */
VF_CUDA_API int vf_match_template_gray_u8(
    void* context,
    uint64_t generation,
    int search_x, int search_y, int search_width, int search_height,
    const uint8_t* templ, int template_width, int template_height,
    int* out_match, float* out_score);

/*
 * Debug helper for the localization extension: after vf_match_template_gray_u8 has run, copies
 * the packed winning key back so a caller can inspect the raw candidate comparison. Not used by
 * production code paths.
 */
VF_CUDA_API int vf_match_template_debug_key(void* context, unsigned long long* out_key);

/* Diagnostics: copy one prefix plane of the localization scratch back to the host. */
VF_CUDA_API int vf_match_template_debug_planes(
    void* context, int plane, int64_t* out_values, size_t count);

/* Diagnostics: copy the gray search ROI that the localization step computed back to the host. */
VF_CUDA_API int vf_match_template_debug_roi(void* context, uint8_t* out_values, size_t count);

/* Diagnostics: copy the per-column candidate scores and rows of the last localization call. */
VF_CUDA_API int vf_match_template_debug_candidates(
    void* context, double* out_scores, int* out_rows, int count, int* candidate_slots);

/*
 * Optional contour extension: cv2.findContours(binary, mode, CHAIN_APPROX_SIMPLE) equivalence.
 *
 * Input path. The operator reads `width` x `height` pixels at (x, y) of the context's *resident*
 * image and treats every non-zero pixel as foreground, exactly like OpenCV binarizes the padded
 * image with THRESH_BINARY. The caller uploads the binary mask itself with vf_context_upload_u8
 * as a single-channel image (1 byte per pixel); the colour image is never uploaded for this step.
 * A resident colour image is reported as VF_CUDA_UNSUPPORTED because reinterpreting it as a mask
 * would silently change the foreground rule.
 *
 * The uploaded region is treated as an isolated image: a one-pixel zero frame is added on the
 * device, so pixels outside the region never influence the result. That is what makes a
 * sub-region call identical to cv2.findContours on the same sub-array.
 *
 * Semantics. Port of tools/contour_reference.py, which is verified point-for-point against
 * cv2.findContours (Suzuki-Abe border following, icvFetchContour with CHAIN_APPROX_SIMPLE):
 *   - VF_CONTOURS_RETR_LIST     == cv2.RETR_LIST
 *   - VF_CONTOURS_RETR_EXTERNAL == cv2.RETR_EXTERNAL
 *   - contour order: OpenCV reports the flat list in *reverse discovery order*; this operator
 *     already returns that order.
 *   - point coordinates: 0-based within the requested region (add x/y for image coordinates).
 *   - deterministic: one serialized thread performs the raster scan and the border traces, no
 *     atomics are used, and the auxiliary kernels write every output element from exactly one
 *     thread. The same resident mask and region therefore always produce the same bytes.
 *
 * Output layout (two calls, mirroring vf_roi_batch_create / vf_roi_batch_download_u8):
 *   vf_find_contours_u8() runs the trace into context-owned device scratch and reports how many
 *     contours and how many points the result holds.
 *   vf_find_contours_download() copies that result to host buffers:
 *     out_offsets: int32[contour_count + 1]; contour j owns point indices
 *                  [out_offsets[j], out_offsets[j + 1]). Use those as *point* indices into
 *                  out_points, which holds point pairs.
 *     out_points:  int32[2 * point_count] as (x, y) pairs; point_capacity counts pairs.
 *   offset_capacity must be at least contour_count + 1 and point_capacity at least point_count;
 *   a short buffer is rejected with VF_CUDA_INVALID_ARGUMENT instead of being filled partially.
 *   The scratch keeps the most recent call only, and a download is rejected when the resident
 *   image changed (generation) after the trace ran.
 */
enum VisionFlowContourMode {
    VF_CONTOURS_RETR_EXTERNAL = 0,
    VF_CONTOURS_RETR_LIST = 1
};

VF_CUDA_API int vf_find_contours_u8(
    void* context,
    uint64_t generation,
    int x, int y, int width, int height, int mode,
    int* out_contour_count, int* out_point_count);

VF_CUDA_API int vf_find_contours_download(
    void* context,
    int32_t* out_offsets, int offset_capacity,
    int32_t* out_points, int point_capacity);

VF_CUDA_API int vf_preprocess_401_2_u8(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride,
    int gaussian_kernel_size,
    int adaptive_block_size, float adaptive_c,
    int max_value, int invert);

/*
 * Optional exact-median extension: the bit-exact NumPy result of ``np.median`` for float32 input.
 *
 * The detector that motivates this export spends most of its time in ``np.median(residual)`` and in
 * the median absolute deviation around it, so the median is the highest-value device step there.
 *
 * Semantics. Every float32 is mapped to a monotone-orderable uint32 key (positives flip the sign
 * bit, negatives invert every bit), so unsigned integer order is exactly float order, including
 * negatives, zeros, subnormals and infinities. The keys are radix-sorted on the device with
 * ``cub::DeviceRadixSort``; only the one or two middle keys cross PCIe, and the float32 average for
 * an even count (add, then divide by two, both in float32 -- NumPy averages in the input dtype)
 * is computed on the host. No device-side floating-point arithmetic is involved, so the result
 * cannot drift from the reference for any finite or infinite input.
 *
 * ``values`` is a host float32 array of ``count`` elements and is only read, never written.
 * ``count`` must be positive and must fit in a signed 32-bit integer (the radix-sort offset type).
 * A NaN anywhere in ``values`` returns NaN, mirroring NumPy's ``_median_nancheck``; note that
 * ``NaN == NaN`` is false, so callers comparing against ``np.median`` must use a NaN-aware test
 * for that case only.
 *
 * The operation is deterministic: identical input bytes always produce identical output bytes.
 * Deviceless callers that want to restart the step on the CPU reference should probe this export
 * and fall back when it is missing.
 */
VF_CUDA_API int vf_median_f32(
    void* context,
    const float* values,
    long long count,
    float* out_median);

/*
 * Optional float32 Gaussian extension: the separable blur the 202-CS-SN-1 detector builds its
 * background with, moved off the host so the residual can stay on the device.
 *
 * Semantics. Single-channel float32, separable horizontal-then-vertical convolution with
 * reflect101 borders and float32 accumulation, using the same coefficients OpenCV's
 * cv2.getGaussianKernel(ksize, 0.0, CV_32F) produces (the 0.3*((ksize-1)*0.5-1)+0.8 sigma for
 * ksize > 9, OpenCV's fixed small-kernel table for ksize 3, 5, 7 and 9 - SMALL_GAUSSIAN_SIZE is 9
 * in OpenCV 5.x). Coefficients were checked bit-for-bit against cv2.getGaussianKernel for every
 * supported size, so the difference against cv2.GaussianBlur(src, (ksize, ksize), 0.0) comes only
 * from the summation order inside OpenCV's SIMD filter:
 *
 *   Verified tolerance (tools/gaussian_f32_equivalence.py,
 *   outputs_validation/cnr_profile/gaussian_f32_equivalence.txt):
 *     - input values in [0, 255] (the detector's gray/residual domain), any supported ksize:
 *       max |device - cv2| <= 2.0e-4 (measured 9.2e-5), mean <= 2.0e-5 (measured 1.1e-5).
 *     - the error scales with the magnitude of the source, not with the image size: it stayed
 *       below 5.0e-7 of the source range (measured 3.5e-7 on a +-1000 float32 source).
 *   The 202-CS-SN-1 result was then compared against the cv2.GaussianBlur reference on 39 scenes /
 *   78 defects (tools/gaussian_202_matrix.py,
 *   outputs_validation/cnr_profile/gaussian_202_final_output_matrix.txt): PASS/NG, defect count,
 *   every defect's bbox, area and confidence, the candidate mask and every other metadata field
 *   were identical in all 39 scenes. The only fields that moved were the residual-derived
 *   diagnostic floats reported inside the defect metadata - residual_median <= 1.5e-5,
 *   mad <= 7.6e-6, robust_noise_sigma <= 1.1e-5, residual_threshold <= 3.4e-5 absolute - which is
 *   the tolerance above appearing in the reported numbers, not in any detection decision.
 *
 * Supported kernel sizes. Exactly the odd sizes in [3, 127] - all 63 of them are covered by the
 * tolerance evidence above. Anything else (even, below 3, or above 127) is rejected: sizes below 3
 * with VF_CUDA_INVALID_ARGUMENT, other unverified sizes with VF_CUDA_UNSUPPORTED, so a caller can
 * restart that step on the CPU reference instead of receiving an unvalidated result.
 *
 * Strides are byte counts. `src` and `dst` are host pointers to single-channel float32 images and
 * both must be at least `width * 4` bytes per row. The result is deterministic: identical input
 * bytes and kernel size always produce identical output bytes.
 *
 * The scratch buffers (uploaded source rectangle, horizontal intermediate, packed result) are owned
 * by the context, grow-only, and separate from the plan scratch, so a call never disturbs a
 * compiled plan. Calls are serialized like every other context export.
 */
VF_CUDA_API int vf_gaussian_blur_f32(
    void* context,
    const float* src, int width, int height, int src_stride,
    float* dst, int dst_stride,
    int kernel_size);

/*
 * Optional rectangle variant of the float32 Gaussian: the same operator restricted to
 * `width` x `height` pixels at (x, y) of a wider `src_width` x `src_height` host image.
 *
 * The rectangle is treated as an isolated image: borders reflect inside it and pixels outside it
 * are never read, so the result equals cv2.GaussianBlur(src[y:y+height, x:x+width], (ksize, ksize),
 * 0.0) and the caller can blur a sub-window without uploading the whole plane. `src_stride` is the
 * byte stride of the full source, `dst_stride` the byte stride of the `width` x `height`
 * single-channel float32 result. Validation, tolerance and ksize rules are identical to
 * vf_gaussian_blur_f32.
 */
VF_CUDA_API int vf_gaussian_blur_f32_roi(
    void* context,
    const float* src, int src_width, int src_height, int src_stride,
    int x, int y, int width, int height,
    float* dst, int dst_stride,
    int kernel_size);

#ifdef __cplusplus
}
#endif

#endif
