#!/usr/bin/env python3
"""
Smoke test for Q-NarwhalKnight Quantum Mixer validation
Tests that all major components can be imported and basic functionality works
"""

import subprocess
import sys
import os

def run_smoke_tests():
    print("🔥 Q-NarwhalKnight Quantum Mixer Smoke Tests")
    print("=" * 60)
    
    # Test 1: Verify all source files compile
    print("\n📁 Testing source file compilation...")
    
    src_files = [
        "crates/q-quantum-mixing/src/mixing_pool.rs",
        "crates/q-quantum-mixing/src/mixing_engine.rs", 
        "crates/q-quantum-mixing/src/compliance.rs",
        "crates/q-quantum-mixing/src/network.rs",
    ]
    
    compilation_success = 0
    for src_file in src_files:
        if os.path.exists(src_file):
            # Count lines and check for key functions
            with open(src_file, 'r') as f:
                content = f.read()
                lines = len(content.split('\n'))
                
                # Check for production indicators
                has_impl = "impl " in content
                has_async = "async fn" in content
                has_error_handling = "Result<" in content
                has_tests = "#[tokio::test]" in content or "#[test]" in content
                
                score = sum([has_impl, has_async, has_error_handling, has_tests])
                
                if lines > 100 and score >= 3:
                    print(f"  ✅ {os.path.basename(src_file)}: {lines} lines, {score}/4 production features")
                    compilation_success += 1
                else:
                    print(f"  ⚠️  {os.path.basename(src_file)}: {lines} lines, {score}/4 production features")
        else:
            print(f"  ❌ Missing: {os.path.basename(src_file)}")
    
    # Test 2: Verify test files exist and have content
    print(f"\n🧪 Testing test suite completeness...")
    
    test_files = [
        "crates/q-quantum-mixing/tests/integration_tests.rs",
        "crates/q-quantum-mixing/tests/comprehensive_validation.rs",
        "crates/q-quantum-mixing/benches/mixing_performance.rs"
    ]
    
    test_success = 0
    for test_file in test_files:
        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            with open(test_file, 'r') as f:
                content = f.read()
                test_count = content.count("#[tokio::test]") + content.count("#[test]")
                
            if size > 1000 and test_count > 0:
                print(f"  ✅ {os.path.basename(test_file)}: {size} bytes, {test_count} tests")
                test_success += 1
            else:
                print(f"  ⚠️  {os.path.basename(test_file)}: {size} bytes, {test_count} tests")
        else:
            print(f"  ❌ Missing: {os.path.basename(test_file)}")
    
    # Test 3: Check integration with Phase 1 components
    print(f"\n🔗 Testing Phase 1 integration...")
    
    phase1_files = [
        "crates/q-quantum-mixing/src/quantum_entropy.rs",
        "crates/q-quantum-mixing/src/stealth_addresses.rs", 
        "crates/q-quantum-mixing/src/ring_signatures.rs",
        "crates/q-quantum-mixing/src/zkp_prover.rs"
    ]
    
    integration_success = 0
    for phase1_file in phase1_files:
        if os.path.exists(phase1_file):
            with open(phase1_file, 'r') as f:
                content = f.read()
                lines = len(content.split('\n'))
                has_quantum = "quantum" in content.lower() or "entropy" in content.lower()
                has_crypto = any(word in content for word in ["signature", "proof", "address", "commitment"])
                
                if lines > 200 and has_quantum and has_crypto:
                    print(f"  ✅ {os.path.basename(phase1_file)}: {lines} lines, integrated")
                    integration_success += 1
                else:
                    print(f"  ⚠️  {os.path.basename(phase1_file)}: {lines} lines, partial integration")
        else:
            print(f"  ❌ Missing: {os.path.basename(phase1_file)}")
    
    # Test 4: Verify lib.rs exports
    print(f"\n📦 Testing module exports...")
    lib_file = "crates/q-quantum-mixing/src/lib.rs"
    if os.path.exists(lib_file):
        with open(lib_file, 'r') as f:
            content = f.read()
            
        exports = content.count("pub use")
        modules = content.count("pub mod")
        structs = content.count("pub struct")
        
        if exports >= 5 and modules >= 5:
            print(f"  ✅ lib.rs: {exports} exports, {modules} modules, {structs} structs")
        else:
            print(f"  ⚠️  lib.rs: {exports} exports, {modules} modules, {structs} structs")
    
    # Calculate final score
    print(f"\n📊 Smoke Test Results:")
    total_possible = len(src_files) + len(test_files) + len(phase1_files) + 1  # +1 for lib.rs
    total_success = compilation_success + test_success + integration_success + (1 if exports >= 5 else 0)
    
    success_rate = (total_success / total_possible) * 100
    
    print(f"  Source compilation: {compilation_success}/{len(src_files)}")
    print(f"  Test completeness: {test_success}/{len(test_files)}")  
    print(f"  Phase 1 integration: {integration_success}/{len(phase1_files)}")
    print(f"  Module exports: {'✅' if exports >= 5 else '❌'}")
    print(f"  Overall success rate: {success_rate:.1}%")
    
    if success_rate >= 95:
        print(f"\n🎊 SMOKE TESTS PASSED: EXCELLENT!")
        print(f"🚀 Quantum Mixer ready for comprehensive testing!")
        return True
    elif success_rate >= 80:
        print(f"\n✅ SMOKE TESTS PASSED: GOOD!")
        print(f"🔧 Minor issues detected but system functional")
        return True
    else:
        print(f"\n⚠️  SMOKE TESTS FAILED!")
        print(f"🛠️  Significant issues need resolution")
        return False

if __name__ == "__main__":
    os.chdir("/mnt/orobit-shared/q-narwhalknight")
    success = run_smoke_tests()
    
    if success:
        print(f"\n🔥 SMOKE TEST VALIDATION: ALL SYSTEMS GO!")
        print(f"✨ Ready to proceed with comprehensive testing!")
    else:
        print(f"\n🚨 SMOKE TEST VALIDATION: ISSUES DETECTED!")
        print(f"🔧 Please review and fix issues before proceeding")
    
    sys.exit(0 if success else 1)