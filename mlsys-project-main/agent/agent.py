import json
import os
import re
import subprocess
from pathlib import Path
from llm.openai_client import client

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "agent" / "prompts"
STATE_FILE = ROOT / "output.json"
FIRST_ITERATION_METRICS_FILE = "/target/target_spec.json"
BENCHMARKS_DIR = ROOT / "benchmarks"

MAX_RETRIES = 2
MAX_ITERATIONS = 5

class ProfilingAgent:
    def __init__(self):
        self.state = self.load_state()
        self.benchmark = "gemm"

    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "iteration": 0,
            "code_version": 0,
            "current_version": 0,
            "metrics_history": [],
            "analysis_history": [],
            "recommended_metrics_history": [],
            "new_benchmarks": [],
            "error_history": [],
            "current_bottleneck": None,
            "done": False
        }

    def save_state(self):
        os.makedirs(str(STATE_FILE.parent), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_prompt(self, file_name: str) -> str:
        prompt_path = PROMPT_DIR / file_name
        return prompt_path.read_text(encoding="utf-8")

    def get_current_benchmark_name(self) -> str:
        if self.state["current_version"] == 0:
            return self.benchmark
        return f"{self.benchmark}_v{self.state['current_version']}"

    def run_profile(self, metrics=None):
        benchmark_name = self.get_current_benchmark_name()
        cmd = ["python", str(ROOT / "runner" / "run.py"), "--benchmark", benchmark_name, "--profile"]
        if metrics:
            if isinstance(metrics, list):
                metrics = ",".join(metrics)
            cmd.extend(["--metrics", metrics])
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Profiling failed: {result.stderr}")
        return result.stdout + result.stderr

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        return text.strip()

    def load_first_iteration_metrics(self) -> list[str]:
        if not FIRST_ITERATION_METRICS_FILE.exists():
            return []
        with open(FIRST_ITERATION_METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", [])
        if not isinstance(metrics, list) or not all(isinstance(m, str) and m.strip() for m in metrics):
            return []
        return [m.strip() for m in metrics]

    def validate_cuda_code(self, code: str) -> list[str]:
        warnings = []
        lines = code.split('\n')
        has_sm_count = False
        has_clock64 = False
        has_grid_launch = False
        grid_size_ok = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#'):
                continue

            if 'cudaDevAttrMultiProcessorCount' in line or 'sm_count' in line.lower():
                has_sm_count = True

            if 'clock64()' in line:
                has_clock64 = True
                if '__global__' not in line and '__device__' not in line:
                    prev_kernel = False
                    for j in range(max(0, i-10), i):
                        if '__global__' in lines[j]:
                            prev_kernel = True
                            break
                    if not prev_kernel:
                        warnings.append(f"WARNING: clock64() at line {i+1} may not be inside kernel")

            if '<<<' in line and '>>>' in line:
                has_grid_launch = True
                if 'sm_count' in line.lower() or '*' in line:
                    grid_size_ok = True
                if re.search(r'<<<\s*\d+\s*,', line):
                    match = re.search(r'<<<\s*(\d+)\s*,', line)
                    if match:
                        blocks = int(match.group(1))
                        if blocks < 32:
                            warnings.append(f"WARNING: Grid size too small ({blocks} blocks). Must use sm_count*4 blocks.")

            if 'double' in line and ('float ' in line or 'double ' in line):
                if 'float ' in line and 'float*' not in line and 'float[' not in line:
                    warnings.append(f"WARNING: Using float instead of double at line {i+1}")

        if not has_sm_count:
            warnings.append("WARNING: No SM count query detected. Must use cudaDeviceGetAttribute to query SM count.")
        if not has_grid_launch:
            warnings.append("WARNING: No kernel launch configuration found.")
        elif not grid_size_ok:
            warnings.append("WARNING: Grid size may be too small. Must use sm_count*4 blocks.")
        if not has_clock64:
            warnings.append("WARNING: No clock64() timing found. Must use clock64() inside kernel for accurate timing.")

        return warnings

    def analyze_metrics(self, profile_output, error_context=None, retry: int = 0):
        if self.state["iteration"] == 1:
            baseline_metrics = self.load_first_iteration_metrics()
            if baseline_metrics:
                metrics_instruction = (
                    "Use the baseline metrics for the first iteration from the metrics file: "
                    + ", ".join(baseline_metrics)
                    + "."
                )
            else:
                metrics_instruction = ""
        else:
            metrics_instruction = (
                "This is not the first iteration. Analyze the output and recommend a focused set of ncu metrics"
                " for the next profiling pass. Include the recommended metrics in the JSON response."
            )

        template = self.load_prompt("analyze_metrics.txt")
        prompt = template.format(
            iteration=self.state["iteration"],
            target_metric=self.state.get("current_bottleneck", "unknown"),
            profile_output=profile_output,
            metrics_instruction=metrics_instruction,
        )
        if error_context:
            prompt += f"\nThe previous LLM response or execution had an issue:\n{error_context}\nPlease correct it and return valid JSON only."

        response = client.chat.completions.create(
            model=os.getenv("BASE_MODEL", ""),
            messages=[{"role": "user", "content": prompt}]
        )
        analysis = response.choices[0].message.content
        json_str = self._extract_json(analysis)
        try:
            analysis_json = json.loads(json_str)
        except json.JSONDecodeError as exc:
            error_text = f"JSON decode error: {exc}\nResponse:\n{analysis}"
            self.state["error_history"].append({
                "iteration": self.state["iteration"],
                "type": "json_parse",
                "message": error_text
            })
            if retry < MAX_RETRIES:
                return self.analyze_metrics(profile_output, error_context=error_text, retry=retry + 1)
            return {"bottleneck": "unknown", "key_metrics": analysis, "new_benchmark_description": "No new benchmark suggested"}

        if not analysis_json.get("new_benchmark_description"):
            analysis_json["new_benchmark_description"] = "No new benchmark suggested"
        return analysis_json

    def _parse_metrics(self, metrics):
        if not metrics:
            return []
        if isinstance(metrics, list):
            return [str(m).strip() for m in metrics if str(m).strip()]
        if isinstance(metrics, str):
            return [m.strip() for m in metrics.split(",") if m.strip()]
        return []

    def generate_new_benchmark(self, description, error_context=None, retry: int = 0):
        target_name = f"Target {self.state['iteration']}: {description}" if description else "GPU compute and memory throughput profiling"
        prompt_template = self.load_prompt("generate_benchmark.txt")
        prompt = prompt_template.format(
            target_name=target_name,
            description=description or "Generate a CUDA benchmark that maximizes sm__throughput and gpu__compute_memory_throughput metrics."
        )
        if error_context:
            prompt += f"\nThe previous code generation or profiling step failed with this error:\n{error_context}\nPlease fix the CUDA code and return valid CUDA source only."

        response = client.chat.completions.create(
            model=os.getenv("BASE_MODEL", ""),
            messages=[{"role": "user", "content": prompt}]
        )
        new_code = response.choices[0].message.content
        code_text = new_code
        if "```cuda" in new_code:
            start = new_code.find("```cuda") + 7
            end = new_code.find("```", start)
            if end == -1:
                end = len(new_code)
            code_text = new_code[start:end].strip()
        elif "```" in new_code:
            start = new_code.find("```") + 3
            end = new_code.find("```", start)
            if end == -1:
                end = len(new_code)
            code_text = new_code[start:end].strip()

        if not code_text.strip():
            error_text = f"Generated code was empty. Raw response:\n{new_code}"
            self.state["error_history"].append({
                "iteration": self.state["iteration"],
                "type": "code_generation",
                "message": error_text
            })
            if retry < MAX_RETRIES:
                return self.generate_new_benchmark(description, error_context=error_text, retry=retry + 1)
            raise RuntimeError(error_text)

        validation_warnings = self.validate_cuda_code(code_text)
        if validation_warnings:
            print(f"[CodeGen] CUDA code validation warnings: {len(validation_warnings)}")
            for w in validation_warnings:
                print(f"  - {w}")
            if retry < MAX_RETRIES:
                error_context = "\n".join(validation_warnings)
                return self.generate_new_benchmark(description, error_context=error_context, retry=retry + 1)

        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        new_version = self.state["code_version"] + 1
        new_code_path = BENCHMARKS_DIR / f"{self.benchmark}_v{new_version}.cu"
        with open(new_code_path, 'w') as f:
            f.write(code_text)
        self.state["code_version"] = new_version
        self.state["current_version"] = new_version
        print(f"[CodeGen] Generated {new_code_path}")
        return new_version

    def iterate(self):
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

        while not self.state["done"] and self.state["iteration"] < MAX_ITERATIONS:
            self.state["iteration"] += 1
            print(f"\n{'='*60}")
            print(f"Iteration {self.state['iteration']}/{MAX_ITERATIONS}")
            print(f"{'='*60}")

            metrics = None
            if self.state["iteration"] > 1 and self.state.get("recommended_metrics_history"):
                last_metrics = self.state["recommended_metrics_history"][-1]
                if last_metrics:
                    metrics = last_metrics
                    print(f"[Pipeline] Using recommended metrics: {metrics}")

            try:
                if self.state["iteration"] == 1 or self.state["current_version"] == 0:
                    self.generate_new_benchmark("")
                profile_output = self.run_profile(metrics=metrics)
                print(f"[Pipeline] Profiling output:\n{profile_output}")
            except RuntimeError as exc:
                error_info = str(exc)
                self.state["error_history"].append({
                    "iteration": self.state["iteration"],
                    "type": "profile",
                    "message": error_info
                })
                print(f"[Pipeline] Profile failed: {error_info}")
                last_description = self.state["new_benchmarks"][-1] if self.state["new_benchmarks"] else "Generate a CUDA benchmark program for GPU profiling."
                try:
                    self.generate_new_benchmark(last_description, error_context=error_info)
                    self.state["new_benchmarks"].append(last_description)
                    self.save_state()
                    continue
                except RuntimeError as gen_exc:
                    print(f"[Pipeline] Failed to regenerate after profile error: {gen_exc}")
                    break

            self.state["metrics_history"].append(profile_output)

            analysis = self.analyze_metrics(profile_output)
            self.state["analysis_history"].append(analysis)
            self.state["current_bottleneck"] = analysis["bottleneck"]

            print(f"\n[Analysis] Bottleneck: {analysis['bottleneck']}")
            print(f"[Analysis] Key Metrics: {analysis['key_metrics']}")
            print(f"[Analysis] Recommended Metrics: {analysis.get('recommended_metrics')}")
            print(f"[Analysis] Next Benchmark: {analysis['new_benchmark_description']}")

            if not analysis["new_benchmark_description"] or analysis["new_benchmark_description"] == "No new benchmark suggested":
                self.state["done"] = True
                print("[Pipeline] No further improvements suggested, finishing.")
                break

            recommended_metrics = self._parse_metrics(analysis.get("recommended_metrics"))
            self.state.setdefault("recommended_metrics_history", []).append(recommended_metrics)
            self.state["analysis_history"][-1]["recommended_metrics"] = recommended_metrics

            try:
                self.generate_new_benchmark(analysis["new_benchmark_description"])
                self.state["new_benchmarks"].append(analysis["new_benchmark_description"])
            except RuntimeError as exc:
                error_info = str(exc)
                self.state["error_history"].append({
                    "iteration": self.state["iteration"],
                    "type": "benchmark_generation",
                    "message": error_info
                })
                print(f"[Pipeline] Benchmark generation failed: {error_info}")
                continue

            self.save_state()

        print(f"\n{'='*60}")
        print(f"Agent finished after {self.state['iteration']} iterations.")
        print(f"{'='*60}")

if __name__ == "__main__":
    agent = ProfilingAgent()
    agent.iterate()
