#!/usr/bin/env python3
"""
SIMD Activation Diagnostic Script
==================================

Purpose: Identify why SIMD code paths are not being activated in LUT-GEMM kernels.
Scans CMakeLists.txt, lut_gemm.cpp, and runtime checks to understand the full picture.

Output: PHASE3_DIAGNOSTIC_REPORT.json with actionable findings
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class SIMDDiagnostics:
    """Comprehensive SIMD activation diagnostics"""
    
    def __init__(self, workspace_root: str):
        self.root = Path(workspace_root)
        self.findings = {
            "timestamp": None,
            "simd_compilation_flags": {},
            "simd_runtime_checks": {},
            "compute_path_configuration": {},
            "issues_found": [],
            "recommendations": [],
            "estimated_speedup": 0.0
        }
    
    def check_cmake_flags(self) -> Dict[str, Any]:
        """Examine CMakeLists.txt for SIMD compilation flags"""
        cmake_path = self.root / "CMakeLists.txt"
        findings = {
            "avx2_flag": False,
            "avx512_flag": False,
            "march_native": False,
            "flags_list": [],
            "file_path": str(cmake_path),
            "content_snippet": None
        }
        
        if not cmake_path.exists():
            findings["error"] = "CMakeLists.txt not found"
            return findings
        
        content = cmake_path.read_text()
        
        # Look for SIMD flags
        patterns = {
            "avx2": r"(?:-mavx2|-DAVX2|march=\w*avx2)",
            "avx512": r"(?:-mavx512f|-DAVX512|march=\w*avx512)",
            "march_native": r"-march=native|${NATIVE_MARCH_FLAG}"
        }
        
        for flag, pattern in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                findings[flag.replace("-", "_")] = True
                findings["flags_list"].append(flag.upper())
        
        # Extract the actual compile_options line
        options_match = re.search(
            r"target_compile_options\(.*?-\w+.*?\)",
            content,
            re.DOTALL
        )
        if options_match:
            findings["content_snippet"] = options_match.group(0)[:200]
        
        return findings
    
    def check_lut_gemm_config(self) -> Dict[str, Any]:
        """Check LUT-GEMM configuration and runtime checks"""
        cpp_path = self.root / "src" / "inference" / "lut_gemm.cpp"
        findings = {
            "file_path": str(cpp_path),
            "use_avx512_gather_config": False,
            "compute_path_conditions": [],
            "constructor_found": False,
            "error": None
        }
        
        if not cpp_path.exists():
            findings["error"] = f"lut_gemm.cpp not found at {cpp_path}"
            return findings
        
        content = cpp_path.read_text()
        
        # Check for config_.use_avx512_gather
        if "config_.use_avx512_gather" in content:
            findings["use_avx512_gather_config"] = True
        
        # Find constructor where it's set
        constructor_match = re.search(
            r"LookupTableGEMM::LookupTableGEMM\([^}]+?\)",
            content,
            re.DOTALL
        )
        if constructor_match:
            findings["constructor_found"] = True
            constructor_text = constructor_match.group(0)
            if "use_avx512_gather" in constructor_text:
                findings["set_in_constructor"] = True
                findings["constructor_snippet"] = constructor_text[:300]
        
        # Find compute path selection logic
        compute_match = re.search(
            r"#if defined\(__AVX512.*?\n.*?#endif",
            content,
            re.DOTALL
        )
        if compute_match:
            findings["compute_path_found"] = True
            findings["compute_path_snippet"] = compute_match.group(0)[:400]
        
        # Check for any runtime CPU detection
        if "#include <cpuid.h>" in content or "cpuid" in content.lower():
            findings["cpu_detection_enabled"] = True
        
        return findings
    
    def check_runtime_behavior(self) -> Dict[str, Any]:
        """Check for runtime behavior issues"""
        findings = {
            "diagnostic_logs": None,
            "runtime_checks_identified": [],
            "potential_issues": []
        }
        
        # Look for LOG statements showing which path is taken
        cpp_path = self.root / "src" / "inference" / "lut_gemm.cpp"
        if cpp_path.exists():
            content = cpp_path.read_text()
            
            # Check for logging
            if "LOG(INFO)" in content or "VLOG" in content:
                findings["has_diagnostic_logging"] = True
                log_matches = re.findall(
                    r'LOG\([^)]+\)\s*<<[^;]+',
                    content
                )
                if log_matches:
                    findings["sample_logs"] = log_matches[:3]
        
        return findings
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run all diagnostic checks"""
        self.findings["cmake_analysis"] = self.check_cmake_flags()
        self.findings["lut_gemm_analysis"] = self.check_lut_gemm_config()
        self.findings["runtime_analysis"] = self.check_runtime_behavior()
        
        # Generate recommendations based on findings
        self._generate_recommendations()
        
        return self.findings
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        cmake = self.findings.get("cmake_analysis", {})
        lut = self.findings.get("lut_gemm_analysis", {})
        
        issues = []
        recommendations = []
        estimated_speedup = 1.0
        
        # Issue 1: AVX2/AVX-512 not in CMakeLists
        if not cmake.get("avx2_flag") and not cmake.get("march_native"):
            issues.append({
                "severity": "CRITICAL",
                "issue": "SIMD flags not found in CMakeLists.txt",
                "impact": "Compiler may default to scalar code even if CPU supports SIMD",
                "probability": "HIGH"
            })
            recommendations.append({
                "priority": 1,
                "action": "Add -march=native or explicit -mavx2/-mavx512f to CMakeLists.txt",
                "estimated_time_min": 10,
                "estimated_speedup": 2.0,
                "code_change": """
# In CMakeLists.txt, find target_compile_options() and add:
if(MSVC)
    target_compile_options(ryot_inference PRIVATE /arch:AVX2)
else()
    target_compile_options(ryot_inference PRIVATE -march=native)
endif()
                """
            })
            estimated_speedup *= 2.0
        
        # Issue 2: AVX-512 config flag management
        if lut.get("use_avx512_gather_config") and not lut.get("set_in_constructor"):
            issues.append({
                "severity": "HIGH",
                "issue": "config_.use_avx512_gather exists but not initialized",
                "impact": "AVX-512 path never taken at runtime",
                "probability": "HIGH"
            })
            recommendations.append({
                "priority": 2,
                "action": "Initialize config_.use_avx512_gather in LookupTableGEMM constructor",
                "estimated_time_min": 15,
                "estimated_speedup": 1.3,
                "code_change": """
LookupTableGEMM::LookupTableGEMM(const Config& cfg) : config_(cfg) {
    // Enable AVX-512 if CPU supports it
    #ifdef __AVX512F__
    config_.use_avx512_gather = true;
    LOG(INFO) << "AVX-512 GEMM path enabled";
    #else
    config_.use_avx512_gather = false;
    #endif
}
                """
            })
            estimated_speedup *= 1.3
        
        # Issue 3: Compute path selection logic
        if lut.get("compute_path_snippet"):
            if "compute_scalar" in lut.get("compute_path_snippet", ""):
                issues.append({
                    "severity": "CRITICAL",
                    "issue": "Scalar fallback in main compute path selection",
                    "impact": "Code defaults to scalar when SIMD unavailable, not if disabled",
                    "probability": "MEDIUM"
                })
        
        # Issue 4: Missing runtime CPU detection
        if not lut.get("cpu_detection_enabled"):
            issues.append({
                "severity": "MEDIUM",
                "issue": "No runtime CPU capability detection",
                "impact": "SIMD flags checked at compile time only",
                "probability": "LOW"
            })
            recommendations.append({
                "priority": 3,
                "action": "Add runtime CPUID check for maximum capability",
                "estimated_time_min": 30,
                "estimated_speedup": 1.1,
                "code_change": """
bool has_avx512() {
    int eax, ebx, ecx, edx;
    __cpuid(eax, ebx, ecx, edx, 7, 0);
    return (ebx & (1 << 16)) != 0;  // AVX-512F
}
                """
            })
        
        self.findings["issues_found"] = issues
        self.findings["recommendations"] = recommendations
        
        # Calculate total estimated speedup
        for rec in recommendations:
            self.findings["estimated_speedup"] *= rec.get("estimated_speedup", 1.0)
    
    def save_report(self, output_path: str):
        """Save findings as JSON report"""
        with open(output_path, 'w') as f:
            json.dump(self.findings, f, indent=2)
        print(f"Report saved to {output_path}")

def main():
    workspace = r"c:\Users\sgbil\Ryot"
    
    print("=" * 70)
    print("SIMD ACTIVATION DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Workspace: {workspace}\n")
    
    diagnostics = SIMDDiagnostics(workspace)
    findings = diagnostics.run_diagnostics()
    
    # Print summary
    print("ISSUES FOUND:")
    for issue in findings.get("issues_found", []):
        print(f"  [{issue['severity']}] {issue['issue']}")
        print(f"    Impact: {issue['impact']}")
        print(f"    Probability: {issue['probability']}\n")
    
    print("RECOMMENDATIONS:")
    for rec in findings.get("recommendations", []):
        print(f"  Priority {rec['priority']}: {rec['action']}")
        print(f"    Time: {rec['estimated_time_min']} min | Speedup: {rec['estimated_speedup']}x\n")
    
    print(f"ESTIMATED TOTAL SPEEDUP: {findings['estimated_speedup']}x")
    print(f"FROM: 0.42 tok/s → TO: {0.42 * findings['estimated_speedup']:.2f} tok/s")
    
    # Save JSON report
    output_path = os.path.join(workspace, "PHASE3_DIAGNOSTIC_REPORT.json")
    diagnostics.save_report(output_path)
    print(f"\nDetailed report: {output_path}")

if __name__ == "__main__":
    main()
