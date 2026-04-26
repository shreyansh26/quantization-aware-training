import os

if os.environ.get("QAT_PATCH_VLLM_W4A8_FP8_VIEW") == "1":
    try:
        from qat.eval.vllm_compat import patch_vllm_w4a8_fp8_scale_view

        patch_vllm_w4a8_fp8_scale_view()
    except Exception:
        pass
