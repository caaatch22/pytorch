# set -ex
# clear

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$LOG_DIR}"
# rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# export HIP_VISIBLE_DEVICES=0
# export HIP_HIDDEN_FREE_MEM 500
# export HIP_TRACE_API=1
# export HIP_DB=api+mem+copy
# export HIP_API_BLOCKING=1
# export HIP_LAUNCH_BLOCKING_KERNELS kernel1,kernel2,...
# export HCC_DB 0x48a
# export HCC_SERIALIZE_KERNEL=3
# export HCC_SERIALIZE_COPY=3

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# bash scripts/amd/copy.sh

export PYTORCH_TEST_WITH_ROCM=1

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
# PYTORCH_DIR="/tmp/pytorch"
PYTORCH_DIR="/dockerx/pytorch_rocm"
# PYTORCH_DIR=$(pwd)

cd $PYTORCH_DIR/test
# ls
pwd

# tests
python test_nn.py --verbose \
	TestNNDeviceTypeCUDA.test_conv_cudnn_ndhwc_cuda_float16 \
	TestNNDeviceTypeCUDA.test_conv_cudnn_ndhwc_cuda_float32 2>&1 |
	tee $LOG_DIR/test_conv_cudnn_ndhwc.log

python test_nn.py --verbose \
	TestNNDeviceTypeCUDA.test_convert_conv2d_weight_memory_format_cuda 2>&1 |
	tee $LOG_DIR/test_convert_conv2d_weight_memory_format.log

python test_nn.py --verbose 2>&1 | tee $LOG_DIR/test_nn.log

# python test_nn.py --verbose  TestNNDeviceTypeCUDA.test_conv_cudnn_ndhwc_cuda_float16
# python test_nn.py --verbose  TestNNDeviceTypeCUDA.test_conv_cudnn_ndhwc_cuda_float32
# python test_nn.py --verbose  TestNNDeviceTypeCUDA.test_convert_conv2d_weight_memory_format_cuda

# python test_ops.py --verbose | tee $LOG_DIR/test_ops.log

# pip3 install pandas openpyxl
# python3 /dockerx/pytorch/scripts/amd/run_fft_test.py

# python test_spectral_ops.py --verbose  TestFFTCUDA.test_empty_fft_fft_fft2_cuda_complex128 | tee $LOG_DIR/test_empty_fft_fft_fft2_cuda_complex128.log
# python test_ops.py --verbose TestMathBitsCUDA.test_conj_view_fft_fft2_cuda_complex64 | tee $LOG_DIR/test_conj_view_fft_fft2_cuda_complex64.log

# python /tmp/pytorch/test/test_spectral_ops.py --verbose TestFFTCUDA.test_empty_fft_fft_fft2_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_fft2_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_fft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_fft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_fft_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_fft_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_fft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_fft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_fftn_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_fftn_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_fftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_fftn_cuda_float64 TestFFTCUDA.test_empty_fft_fft_hfft2_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_hfft2_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_hfft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_hfft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_hfft_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_hfft_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_hfft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_hfft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_hfftn_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_hfftn_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_hfftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_hfftn_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ifft2_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_ifft2_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_ifft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ifft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ifft_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_ifft_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_ifft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ifft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ifftn_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_ifftn_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_ifftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ifftn_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ihfft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ihfft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ihfft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ihfft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_ihfftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_ihfftn_cuda_float64 TestFFTCUDA.test_empty_fft_fft_irfft2_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_irfft2_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_irfft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_irfft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_irfft_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_irfft_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_irfft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_irfft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_irfftn_cuda_complex128 TestFFTCUDA.test_empty_fft_fft_irfftn_cuda_complex64 TestFFTCUDA.test_empty_fft_fft_irfftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_irfftn_cuda_float64 TestFFTCUDA.test_empty_fft_fft_rfft2_cuda_float32 TestFFTCUDA.test_empty_fft_fft_rfft2_cuda_float64 TestFFTCUDA.test_empty_fft_fft_rfft_cuda_float32 TestFFTCUDA.test_empty_fft_fft_rfft_cuda_float64 TestFFTCUDA.test_empty_fft_fft_rfftn_cuda_float32 TestFFTCUDA.test_empty_fft_fft_rfftn_cuda_float64 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_fftn_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_hfftn_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ifftn_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_ihfftn_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_irfftn_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfft2_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfft2_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfft_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfft_cuda_float16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfftn_cuda_bfloat16 TestFFTCUDA.test_fft_half_and_bfloat16_errors_fft_rfftn_cuda_float16 TestFFTCUDA.test_fftn_invalid_fft_fftn_cuda_complex64 TestFFTCUDA.test_fftn_invalid_fft_fftn_cuda_float32 TestFFTCUDA.test_fftn_invalid_fft_hfftn_cuda_complex64 TestFFTCUDA.test_fftn_invalid_fft_hfftn_cuda_float32 TestFFTCUDA.test_fftn_invalid_fft_ifftn_cuda_complex64 TestFFTCUDA.test_fftn_invalid_fft_ifftn_cuda_float32 TestFFTCUDA.test_fftn_invalid_fft_ihfftn_cuda_float32 TestFFTCUDA.test_fftn_invalid_fft_irfftn_cuda_complex64 TestFFTCUDA.test_fftn_invalid_fft_irfftn_cuda_float32 TestFFTCUDA.test_fftn_invalid_fft_rfftn_cuda_float32 TestFFTCUDA.test_reference_1d_fft_fft_cuda_complex128 TestFFTCUDA.test_reference_1d_fft_fft_cuda_complex64 TestFFTCUDA.test_reference_1d_fft_fft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_fft_cuda_float64 TestFFTCUDA.test_reference_1d_fft_hfft_cuda_complex128 TestFFTCUDA.test_reference_1d_fft_hfft_cuda_complex64 TestFFTCUDA.test_reference_1d_fft_hfft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_hfft_cuda_float64 TestFFTCUDA.test_reference_1d_fft_ifft_cuda_complex128 TestFFTCUDA.test_reference_1d_fft_ifft_cuda_complex64 TestFFTCUDA.test_reference_1d_fft_ifft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_ifft_cuda_float64 TestFFTCUDA.test_reference_1d_fft_ihfft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_ihfft_cuda_float64 TestFFTCUDA.test_reference_1d_fft_irfft_cuda_complex128 TestFFTCUDA.test_reference_1d_fft_irfft_cuda_complex64 TestFFTCUDA.test_reference_1d_fft_irfft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_irfft_cuda_float64 TestFFTCUDA.test_reference_1d_fft_rfft_cuda_float32 TestFFTCUDA.test_reference_1d_fft_rfft_cuda_float64 TestFFTCUDA.test_reference_nd_fft_fftn_cuda_complex128 TestFFTCUDA.test_reference_nd_fft_fftn_cuda_complex64 TestFFTCUDA.test_reference_nd_fft_fftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_fftn_cuda_float64 TestFFTCUDA.test_reference_nd_fft_hfftn_cuda_complex128 TestFFTCUDA.test_reference_nd_fft_hfftn_cuda_complex64 TestFFTCUDA.test_reference_nd_fft_hfftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_hfftn_cuda_float64 TestFFTCUDA.test_reference_nd_fft_ifftn_cuda_complex128 TestFFTCUDA.test_reference_nd_fft_ifftn_cuda_complex64 TestFFTCUDA.test_reference_nd_fft_ifftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_ifftn_cuda_float64 TestFFTCUDA.test_reference_nd_fft_ihfftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_ihfftn_cuda_float64 TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_complex128 TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_complex64 TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_float64 TestFFTCUDA.test_reference_nd_fft_rfftn_cuda_float32 TestFFTCUDA.test_reference_nd_fft_rfftn_cuda_float64
# python /tmp/pytorch/test/test_ops.py --verbose TestMathBitsCUDA.test_conj_view_fft_fft2_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_fft_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_fftn_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_hfft2_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_hfft_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_hfftn_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_ifft2_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_ifft_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_ifftn_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_irfft2_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_irfft_cuda_complex64 TestMathBitsCUDA.test_conj_view_fft_irfftn_cuda_complex64 TestMathBitsCUDA.test_neg_conj_view_fft_fft2_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_fft_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_fftn_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_hfft2_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_hfft_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_hfftn_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_ifft2_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_ifft_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_ifftn_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_irfft2_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_irfft_cuda_complex128 TestMathBitsCUDA.test_neg_conj_view_fft_irfftn_cuda_complex128 TestMathBitsCUDA.test_neg_view_fft_fft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_fft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_fftn_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_hfft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_hfft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_hfftn_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ifft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ifft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ifftn_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ihfft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ihfft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_ihfftn_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_irfft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_irfft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_irfftn_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_rfft2_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_rfft_cuda_float64 TestMathBitsCUDA.test_neg_view_fft_rfftn_cuda_float64
# python /tmp/pytorch/test/test_ops.py --verbose TestCommonCUDA.test_dtypes_fft_fft2_cuda TestCommonCUDA.test_dtypes_fft_fft_cuda TestCommonCUDA.test_dtypes_fft_fftn_cuda TestCommonCUDA.test_dtypes_fft_hfft2_cuda TestCommonCUDA.test_dtypes_fft_hfft_cuda TestCommonCUDA.test_dtypes_fft_hfftn_cuda TestCommonCUDA.test_dtypes_fft_ifft2_cuda TestCommonCUDA.test_dtypes_fft_ifft_cuda TestCommonCUDA.test_dtypes_fft_ifftn_cuda TestCommonCUDA.test_dtypes_fft_ihfft2_cuda TestCommonCUDA.test_dtypes_fft_ihfft_cuda TestCommonCUDA.test_dtypes_fft_ihfftn_cuda TestCommonCUDA.test_dtypes_fft_irfft2_cuda TestCommonCUDA.test_dtypes_fft_irfft_cuda TestCommonCUDA.test_dtypes_fft_irfftn_cuda TestCommonCUDA.test_dtypes_fft_rfft2_cuda TestCommonCUDA.test_dtypes_fft_rfft_cuda TestCommonCUDA.test_dtypes_fft_rfftn_cuda TestCommonCUDA.test_noncontiguous_samples_fft_fft2_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_fft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_fft_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_fft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_fftn_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_fftn_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_hfft2_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_hfft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_hfft_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_hfft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_hfftn_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_hfftn_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ifft2_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_ifft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ifft_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_ifft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ifftn_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_ifftn_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ihfft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ihfft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_ihfftn_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_irfft2_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_irfft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_irfft_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_irfft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_irfftn_cuda_complex64 TestCommonCUDA.test_noncontiguous_samples_fft_irfftn_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_rfft2_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_rfft_cuda_float32 TestCommonCUDA.test_noncontiguous_samples_fft_rfftn_cuda_float32 TestCommonCUDA.test_out_fft_fft2_cuda TestCommonCUDA.test_out_fft_fft_cuda TestCommonCUDA.test_out_fft_fftn_cuda TestCommonCUDA.test_out_fft_hfft2_cuda TestCommonCUDA.test_out_fft_hfft_cuda TestCommonCUDA.test_out_fft_hfftn_cuda TestCommonCUDA.test_out_fft_ifft2_cuda TestCommonCUDA.test_out_fft_ifft_cuda TestCommonCUDA.test_out_fft_ifftn_cuda TestCommonCUDA.test_out_fft_ihfft2_cuda TestCommonCUDA.test_out_fft_ihfft_cuda TestCommonCUDA.test_out_fft_ihfftn_cuda TestCommonCUDA.test_out_fft_irfft2_cuda TestCommonCUDA.test_out_fft_irfft_cuda TestCommonCUDA.test_out_fft_irfftn_cuda TestCommonCUDA.test_out_fft_rfft2_cuda TestCommonCUDA.test_out_fft_rfft_cuda TestCommonCUDA.test_out_fft_rfftn_cuda TestCommonCUDA.test_out_warning_fft_fft2_cuda TestCommonCUDA.test_out_warning_fft_fft_cuda TestCommonCUDA.test_out_warning_fft_fftn_cuda TestCommonCUDA.test_out_warning_fft_hfft2_cuda TestCommonCUDA.test_out_warning_fft_hfft_cuda TestCommonCUDA.test_out_warning_fft_hfftn_cuda TestCommonCUDA.test_out_warning_fft_ifft2_cuda TestCommonCUDA.test_out_warning_fft_ifft_cuda TestCommonCUDA.test_out_warning_fft_ifftn_cuda TestCommonCUDA.test_out_warning_fft_ihfft2_cuda TestCommonCUDA.test_out_warning_fft_ihfft_cuda TestCommonCUDA.test_out_warning_fft_ihfftn_cuda TestCommonCUDA.test_out_warning_fft_irfft2_cuda TestCommonCUDA.test_out_warning_fft_irfft_cuda TestCommonCUDA.test_out_warning_fft_irfftn_cuda TestCommonCUDA.test_out_warning_fft_rfft2_cuda TestCommonCUDA.test_out_warning_fft_rfft_cuda TestCommonCUDA.test_out_warning_fft_rfftn_cuda TestCommonCUDA.test_variant_consistency_eager_fft_fft2_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_fft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_fft_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_fft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_fftn_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_fftn_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_hfft2_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_hfft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_hfft_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_hfft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_hfftn_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_hfftn_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ifft2_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_ifft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ifft_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_ifft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ifftn_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_ifftn_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ihfft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ihfft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_ihfftn_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_irfft2_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_irfft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_irfft_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_irfft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_irfftn_cuda_complex64 TestCommonCUDA.test_variant_consistency_eager_fft_irfftn_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_rfft2_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_rfft_cuda_float32 TestCommonCUDA.test_variant_consistency_eager_fft_rfftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fft2_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_fft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_fft_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_fft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_fftn_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_fftn_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_hfft2_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_hfft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_hfft_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_hfft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_hfftn_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_hfftn_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ifft2_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_ifft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ifft_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_ifft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ifftn_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_ifftn_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ihfft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ihfft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_ihfftn_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_irfft2_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_irfft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_irfft_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_irfft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_irfftn_cuda_complex64 TestJitCUDA.test_variant_consistency_jit_fft_irfftn_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_rfft2_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_rfft_cuda_float32 TestJitCUDA.test_variant_consistency_jit_fft_rfftn_cuda_float32
# python $PYTORCH_DIR/test/test_ops_gradients.py --verbose TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fft2_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fft_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fftn_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_fftn_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfft2_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfft_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfftn_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_hfftn_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifft2_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifft_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifftn_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ifftn_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ihfft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ihfft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_ihfftn_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfft2_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfft_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfftn_cuda_complex128 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_irfftn_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_rfft2_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_rfft_cuda_float64 TestGradientsCUDA.test_fn_fwgrad_bwgrad_fft_rfftn_cuda_float64 TestGradientsCUDA.test_fn_grad_eig_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_fft2_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_fft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_fft_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_fft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_fftn_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_fftn_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_hfft2_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_hfft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_hfft_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_hfft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_hfftn_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_hfftn_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ifft2_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_ifft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ifft_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_ifft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ifftn_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_ifftn_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ihfft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ihfft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_ihfftn_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_irfft2_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_irfft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_irfft_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_irfft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_irfftn_cuda_complex128 TestGradientsCUDA.test_fn_grad_fft_irfftn_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_rfft2_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_rfft_cuda_float64 TestGradientsCUDA.test_fn_grad_fft_rfftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_fft2_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_fft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_fft_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_fft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_fftn_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_fftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_hfft2_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_hfft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_hfft_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_hfft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_hfftn_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_hfftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ifft2_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_ifft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ifft_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_ifft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ifftn_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_ifftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ihfft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ihfft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_ihfftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_irfft2_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_irfft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_irfft_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_irfft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_irfftn_cuda_complex128 TestGradientsCUDA.test_fn_gradgrad_fft_irfftn_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_rfft2_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_rfft_cuda_float64 TestGradientsCUDA.test_fn_gradgrad_fft_rfftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_fft2_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_fft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_fft_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_fft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_fftn_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_fftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_hfft2_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_hfft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_hfft_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_hfft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_hfftn_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_hfftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ifft2_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_ifft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ifft_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_ifft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ifftn_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_ifftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ihfft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ihfft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_ihfftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_irfft2_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_irfft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_irfft_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_irfft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_irfftn_cuda_complex128 TestGradientsCUDA.test_forward_mode_AD_fft_irfftn_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_rfft2_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_rfft_cuda_float64 TestGradientsCUDA.test_forward_mode_AD_fft_rfftn_cuda_float64

# python $PYTORCH_DIR/test/test_ops_gradients.py --verbose TestGradientsCUDA.test_fn_grad_eig_cuda_complex128

# test_ops_jit might fail if run togther
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_rfftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fft2_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fft_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fftn_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_fftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfft2_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfft_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfftn_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_hfftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifft2_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifft_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifftn_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ifftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ihfft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ihfft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_ihfftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfft2_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfft_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfftn_cuda_complex64
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_irfftn_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_rfft2_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_rfft_cuda_float32
# python /tmp/pytorch/test/test_ops_jit.py --verbose TestJitCUDA.test_variant_consistency_jit_fft_rfftn_cuda_float32
