#!/usr/bin/env python3
"""Kaggle Automation - uses kagglesdk with api_token auth.

The old KaggleApi class has a 401 issue with the CLI and file-based auth.
The new kagglesdk with api_token parameter works correctly.

Handles:
1. Build Kaggle package from local source
2. Push via kagglesdk.save_kernel()
3. Poll status until complete
4. Download output and results.json
5. Analyze results, report pass/fail
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable system proxy for Kaggle API calls (fixes ProxyError on Windows)
_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_proxy_handler)
urllib.request.install_opener(_proxy_handler)

from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest


# ============================================================
# Configuration — change these or set via environment variables
# ============================================================
KAGGLE_API_TOKEN = os.environ.get("KAGGLE_API_KEY", "KGAT_1049b301674a4aea68205e7d2d8594ae")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "wegaza")
KERNEL_SLUG = "gpu-profiling-system-test"
KERNEL_ID = 115545948  # Existing kernel ID — updates use this, creates get auto-assigned
KERNEL_TITLE = "GPU Profiling System Test"
KERNEL_LANGUAGE = "python3"
KERNEL_TYPE = "script"
ENABLE_GPU = True
ENABLE_INTERNET = True
IS_PRIVATE = True


class KaggleAutomator:
    def __init__(self, project_root: str | None = None, api_token: str | None = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.api_token = api_token or KAGGLE_API_TOKEN
        self.kernel_slug = KERNEL_SLUG
        self.client = KaggleClient(api_token=self.api_token)
        self.api = self.client.kernels.kernels_api_client
        print("Kaggle kagglesdk authenticated via api_token")

    # ---- package -------------------------------------------------------

    def prepare_package(self) -> Path:
        """Collect kernel source and metadata for push."""
        print("Preparing kernel files...")

        # Read the kernel source code
        kernel_path = self.project_root / "kaggle_kernel.py"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {kernel_path}")

        kernel_text = kernel_path.read_text(encoding="utf-8")
        print(f"Kernel source: {len(kernel_text)} chars")

        # Collect supporting files to include as dataset sources
        # For now, just push the kernel source
        return kernel_text

    # ---- push ----------------------------------------------------------

    def push_to_kaggle(self, kernel_text: str) -> bool:
        """Push kernel via kagglesdk save_kernel API."""
        print(f"Pushing kernel (id={KERNEL_ID})...")
        try:
            req = ApiSaveKernelRequest()
            req.id = KERNEL_ID
            req.text = kernel_text
            req.language = KERNEL_LANGUAGE
            req.kernel_type = KERNEL_TYPE
            req.is_private = IS_PRIVATE
            req.enable_gpu = ENABLE_GPU
            req.enable_internet = ENABLE_INTERNET

            resp = self.api.save_kernel(req)
            result = resp.to_dict() if hasattr(resp, 'to_dict') else str(resp)
            print(f"Push response: {result}")
            return True
        except Exception as e:
            err = str(e)
            print(f"Push failed: {err[:500]}")
            return False

    # ---- monitor -------------------------------------------------------

    def monitor(self, timeout_min: int = 90) -> Tuple[str, bool]:
        kernel_id = f"{KAGGLE_USERNAME}/{self.kernel_slug}"
        start = time.time()
        last_status = None
        print(f"Monitoring {kernel_id} (timeout {timeout_min} min)...")
        while time.time() - start < timeout_min * 60:
            try:
                resp = self.api.get_kernel(kernel_id)
                # Parse response to get status
                if isinstance(resp, dict):
                    status_text = resp.get("status", "unknown").lower()
                    run_status = resp.get("runStatus", resp.get("run_status", "")).lower()
                else:
                    status_text = str(resp).lower()
                    run_status = ""

                combined = f"{status_text} {run_status}"
                if combined != last_status:
                    print(f"  Status: {combined}")
                    last_status = combined

                if "complete" in combined or "success" in combined:
                    print("Completed!")
                    return combined, True
                if "error" in combined or "failed" in combined or "cancelled" in combined:
                    print(f"Failed: {combined}")
                    return combined, False
            except Exception as e:
                print(f"  Status check error: {e}")
            time.sleep(30)
        print("TIMEOUT")
        return "timeout", False

    # ---- download ------------------------------------------------------

    def download_results(self) -> bool:
        kernel_id = f"{KAGGLE_USERNAME}/{self.kernel_slug}"
        out_dir = self.project_root / "kaggle_output"
        out_dir.mkdir(exist_ok=True)
        print(f"Downloading output to {out_dir}...")
        try:
            resp = self.api.download_kernel_output(kernel_id)
            if resp:
                # Save the response
                out_file = out_dir / "kernel_output.zip"
                if hasattr(resp, 'read'):
                    out_file.write_bytes(resp.read())
                else:
                    out_file.write_bytes(str(resp).encode())
                print(f"Output saved to {out_file}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    # ---- analyse -------------------------------------------------------

    def analyse_results(self, output_dir: str | None = None) -> Dict:
        d = Path(output_dir) if output_dir else self.project_root / "kaggle_output"
        rj = d / "results.json"
        if not rj.exists():
            rj = self.project_root / "results.json"
        if not rj.exists():
            return {"success": False, "error": "results.json not found"}
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
            m = data.get("measurements", {})
            required = ["dram_latency_cycles", "l2_cache_size_mb",
                        "actual_boost_clock_mhz", "sm_count"]
            missing = [k for k in required if k not in m]
            cv = data.get("cross_validation", {})
            passed = sum(1 for v in cv.values() if v is True)
            total = len(cv)
            return {
                "success": len(missing) == 0,
                "measurements": m,
                "missing": missing,
                "cv_passed": passed,
                "cv_total": total,
                "methodology_len": len(data.get("methodology", "")),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- full workflow -------------------------------------------------

    def run(self, max_wait_min: int = 90) -> Dict:
        print("=" * 60)
        print("  Kaggle Automation Workflow")
        print("=" * 60)

        # 1. prepare
        kernel_text = self.prepare_package()

        # 2. push
        if not self.push_to_kaggle(kernel_text):
            print("Push failed")
            return {"success": False, "error": "push_failed"}

        # 3. monitor
        status, ok = self.monitor(timeout_min=max_wait_min)
        if not ok:
            print(f"Kernel ended: {status}")

        # 4. download
        self.download_results()

        # 5. analyse
        result = self.analyse_results()
        print("\n" + "=" * 60)
        print("  FINAL RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        return result


def main():
    automator = KaggleAutomator()
    try:
        result = automator.run()
        sys.exit(0 if result.get("success") else 1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
