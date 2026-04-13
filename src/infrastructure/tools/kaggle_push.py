"""Kaggle push handler — infrastructure layer.

Implements the kaggle_push tool contract. Allows agents to:
1. Push kernel source to Kaggle
2. Optionally monitor execution until complete
3. Download output files (results.json, logs, etc.)

Uses the kagglesdk Python SDK with api_token authentication.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from typing import Any

# Disable system proxy (fixes ProxyError on Windows with Clash)
_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)
urllib.request.install_opener(_proxy_handler)


def kaggle_push_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    """Push kernel to Kaggle and optionally monitor execution.

    Args must match the tool contract input_schema.
    """
    try:
        from kagglesdk import KaggleClient
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiSaveKernelRequest,
            ApiCreateKernelSessionRequest,
            ApiGetKernelSessionStatusRequest,
            ApiDownloadKernelOutputRequest,
        )
    except ImportError:
        return {
            "success": False,
            "error": "kagglesdk not installed. Run: pip install kagglesdk",
        }

    # Extract arguments
    kernel_text = arguments.get("kernel_text", "")
    kernel_id = arguments.get("kernel_id", None)
    kernel_title = arguments.get("kernel_title", "GPU Profiling Test")
    kernel_slug = arguments.get("kernel_slug", "gpu-profiling-test")
    language = arguments.get("language", "python3")
    kernel_type = arguments.get("kernel_type", "notebook")
    enable_gpu = arguments.get("enable_gpu", True)
    enable_internet = arguments.get("enable_internet", True)
    is_private = arguments.get("is_private", True)
    monitor = arguments.get("monitor", False)
    timeout_min = arguments.get("timeout_min", 90)

    if not kernel_text:
        return {"success": False, "error": "kernel_text is required"}

    # Get API token from env or config
    api_token = os.environ.get("KAGGLE_API_KEY", "")
    if not api_token or len(api_token) < 10:
        # Try to read from kaggle.json
        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.isfile(kaggle_json):
            with open(kaggle_json) as f:
                kaggle_config = json.load(f)
            api_token = kaggle_config.get("key", "")
    if not api_token or len(api_token) < 10:
        return {
            "success": False,
            "error": "KAGGLE_API_KEY env var or ~/.kaggle/kaggle.json not configured",
        }

    result: dict[str, Any] = {"success": False}

    try:
        # Authenticate
        client = KaggleClient(api_token=api_token)
        api = client.kernels.kernels_api_client

        # Push kernel
        req = ApiSaveKernelRequest()
        if kernel_id:
            req.id = kernel_id
        req.text = kernel_text
        req.title = kernel_title
        req.slug = kernel_slug
        req.language = language
        req.kernel_type = kernel_type
        req.is_private = is_private
        req.enable_gpu = enable_gpu
        req.enable_internet = enable_internet

        resp = api.save_kernel(req)
        resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else str(resp)
        result["push_response"] = str(resp_dict)[:500]

        # Extract kernel ID from response
        if isinstance(resp_dict, dict):
            result["kernel_id"] = resp_dict.get("id", kernel_id)
        result["kernel_url"] = f"https://www.kaggle.com/code/{kernel_slug}"

        if not monitor:
            result["success"] = True
            result["message"] = "Kernel pushed successfully"
            return result

        # Monitor: create session and wait for completion
        print(f"  [kaggle] Creating kernel session...")
        session_req = ApiCreateKernelSessionRequest()
        session_req.kernel_slug = kernel_slug
        session_req.language = language
        session_req.kernel_type = kernel_type
        session_req.enable_internet = enable_internet
        session_req.machine_shape = "gpu" if enable_gpu else "cpu"

        session_resp = api.create_kernel_session(session_req)
        session_dict = session_resp.to_dict() if hasattr(session_resp, "to_dict") else str(session_resp)
        print(f"  [kaggle] Session created: {str(session_dict)[:300]}")

        # Extract session ID
        if isinstance(session_dict, dict):
            metadata = session_dict.get("metadata", {})
            session_id = metadata.get("kernelSessionId")
            result["session_id"] = session_id
        else:
            result["session_id"] = None

        # Poll status
        print(f"  [kaggle] Monitoring session (timeout: {timeout_min} min)...")
        start = time.time()
        last_status = None
        while time.time() - start < timeout_min * 60:
            try:
                status_req = ApiGetKernelSessionStatusRequest()
                status_req.kernel_slug = kernel_slug
                # user_name needed for status check
                status_req.user_name = "wegaza"  # TODO: get from config
                status_resp = api.get_kernel_session_status(status_req)
                status_dict = status_resp.to_dict() if hasattr(status_resp, "to_dict") else {}
                status = status_dict.get("status", "unknown")
                failure = status_dict.get("failureMessage")

                if status != last_status:
                    print(f"  [kaggle] Status: {status}")
                    last_status = status

                if status.lower() in ("complete", "success"):
                    result["session_status"] = status
                    result["success"] = True
                    break
                elif status.lower() in ("error", "failed", "cancelled"):
                    result["session_status"] = status
                    result["error"] = failure or "Session failed"
                    break
            except Exception as e:
                print(f"  [kaggle] Status check error: {e}")
            time.sleep(30)
        else:
            result["session_status"] = "timeout"
            result["error"] = f"Session timed out after {timeout_min} minutes"

        # Download output if complete
        if result.get("success") and result.get("session_status") in ("complete", "success"):
            try:
                dl_req = ApiDownloadKernelOutputRequest()
                dl_req.owner_slug = "wegaza"  # TODO: get from config
                dl_req.kernel_slug = kernel_slug
                dl_resp = api.download_kernel_output(dl_req)
                if dl_resp:
                    # Save output
                    out_dir = os.path.join(os.getcwd(), "kaggle_output")
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, "kernel_output.zip")
                    if hasattr(dl_resp, "content"):
                        with open(out_file, "wb") as f:
                            f.write(dl_resp.content)
                    elif hasattr(dl_resp, "read"):
                        with open(out_file, "wb") as f:
                            f.write(dl_resp.read())
                    result["output_zip"] = out_file
                    result["message"] = f"Kernel complete. Output saved to {out_file}"
            except Exception as e:
                print(f"  [kaggle] Output download error: {e}")

    except Exception as e:
        result["error"] = str(e)

    return result
