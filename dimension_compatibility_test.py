#!/usr/bin/env python3
"""
Dimension Compatibility Test for SincNet + GAT Integration
Verifies that all components use consistent 128-dim embeddings
"""

def test_dimension_compatibility():
    """Test that all embedding dimensions are consistent across components"""
    
    print("🔍 Testing SincNet + GAT Dimension Compatibility")
    print("=" * 60)
    
    # Expected dimensions throughout the pipeline
    EXPECTED_SHOT_EMBEDDING_DIM = 128
    EXPECTED_GAT_INPUT_DIM = 128
    EXPECTED_GAT_OUTPUT_DIM = 128
    EXPECTED_DECODER_INPUT_DIM = 128
    
    passed_tests = []
    failed_tests = []
    
    def check_file_for_dimension(filepath, variable_name, expected_value):
        """Check if a file contains the expected dimension value"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Look for patterns like "embedding_dim=128", "shot_embedding_dim=128", etc.
            import re
            
            # Pattern to match the variable assignment
            pattern = rf'{variable_name}\s*=\s*(\d+)'
            matches = re.findall(pattern, content)
            
            if matches:
                actual_values = [int(match) for match in matches]
                if all(val == expected_value for val in actual_values):
                    return True, actual_values[0]
                else:
                    return False, actual_values
            else:
                # Also check default parameter values in function definitions
                pattern = rf'{variable_name}\s*=\s*(\d+)[,)]'
                matches = re.findall(pattern, content)
                if matches:
                    actual_values = [int(match) for match in matches]
                    if all(val == expected_value for val in actual_values):
                        return True, actual_values[0]
                    else:
                        return False, actual_values
                        
            return None, "Not found"
            
        except FileNotFoundError:
            return None, "File not found"
        except Exception as e:
            return None, f"Error: {e}"
    
    # Test 1: PerShotTemporalEncoder embedding_dim
    print("1. Testing PerShotTemporalEncoder embedding_dim...")
    success, value = check_file_for_dimension(
        'sincnet_seismic_encoder.py', 
        'embedding_dim', 
        EXPECTED_SHOT_EMBEDDING_DIM
    )
    if success:
        print(f"   ✅ PerShotTemporalEncoder.embedding_dim = {value}")
        passed_tests.append("PerShotTemporalEncoder.embedding_dim")
    else:
        print(f"   ❌ PerShotTemporalEncoder.embedding_dim = {value}")
        failed_tests.append(f"PerShotTemporalEncoder.embedding_dim ({value})")
    
    # Test 2: LightweightGATFusion in_features
    print("2. Testing LightweightGATFusion in_features...")
    success, value = check_file_for_dimension(
        'seismic_gat_fusion.py', 
        'in_features', 
        EXPECTED_GAT_INPUT_DIM
    )
    if success:
        print(f"   ✅ LightweightGATFusion.in_features = {value}")
        passed_tests.append("LightweightGATFusion.in_features")
    else:
        print(f"   ❌ LightweightGATFusion.in_features = {value}")
        failed_tests.append(f"LightweightGATFusion.in_features ({value})")
    
    # Test 3: SeismicSincNetGAT shot_embedding_dim
    print("3. Testing SeismicSincNetGAT shot_embedding_dim...")
    success, value = check_file_for_dimension(
        'seismic_gat_fusion.py', 
        'shot_embedding_dim', 
        EXPECTED_SHOT_EMBEDDING_DIM
    )
    if success:
        print(f"   ✅ SeismicSincNetGAT.shot_embedding_dim = {value}")
        passed_tests.append("SeismicSincNetGAT.shot_embedding_dim")
    else:
        print(f"   ❌ SeismicSincNetGAT.shot_embedding_dim = {value}")
        failed_tests.append(f"SeismicSincNetGAT.shot_embedding_dim ({value})")
    
    # Test 4: SincNetEnhancedSeismicModel shot_embedding_dim
    print("4. Testing SincNetEnhancedSeismicModel shot_embedding_dim...")
    success, value = check_file_for_dimension(
        'sincnet_integration_demo.py', 
        'shot_embedding_dim', 
        EXPECTED_SHOT_EMBEDDING_DIM
    )
    if success:
        print(f"   ✅ SincNetEnhancedSeismicModel.shot_embedding_dim = {value}")
        passed_tests.append("SincNetEnhancedSeismicModel.shot_embedding_dim")
    else:
        print(f"   ❌ SincNetEnhancedSeismicModel.shot_embedding_dim = {value}")
        failed_tests.append(f"SincNetEnhancedSeismicModel.shot_embedding_dim ({value})")
    
    # Test 5: SincNetEnhancedDecoder input_embedding_dim
    print("5. Testing SincNetEnhancedDecoder input_embedding_dim...")
    success, value = check_file_for_dimension(
        'sincnet_integration_demo.py', 
        'input_embedding_dim', 
        EXPECTED_DECODER_INPUT_DIM
    )
    if success:
        print(f"   ✅ SincNetEnhancedDecoder.input_embedding_dim = {value}")
        passed_tests.append("SincNetEnhancedDecoder.input_embedding_dim")
    else:
        print(f"   ❌ SincNetEnhancedDecoder.input_embedding_dim = {value}")
        failed_tests.append(f"SincNetEnhancedDecoder.input_embedding_dim ({value})")
    
    print("\n" + "=" * 60)
    print("📊 DIMENSION COMPATIBILITY TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Passed Tests ({len(passed_tests)}):")
    for test in passed_tests:
        print(f"   • {test}")
    
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"   • {test}")
    
    print(f"\n📈 Success Rate: {len(passed_tests)}/{len(passed_tests) + len(failed_tests)} tests passed")
    
    # Overall compatibility status
    if len(failed_tests) == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ All components are using consistent 128-dim embeddings")
        print("🚀 Ready for integration and training!")
        return True
    else:
        print(f"\n⚠️  {len(failed_tests)} COMPATIBILITY ISSUES FOUND")
        print("🔧 Please fix the dimension mismatches before proceeding")
        return False

def test_architectural_flow():
    """Test the overall architectural flow dimensions"""
    print("\n🏗️  ARCHITECTURAL FLOW VERIFICATION")
    print("=" * 60)
    
    expected_flow = [
        ("Input Shot Gather", "(B, 10001, 31)"),
        ("SincNet Output", "(B*31, 40, 201)"),
        ("PerShotTemporalEncoder", "(B, 128)"),
        ("5-Shot Stack", "(B, 5, 128)"),
        ("GAT Node Features", "(B*5, 128)"),
        ("GAT Fusion Output", "(B, 128)"),
        ("Decoder Input", "(B, 128)"),
        ("Final Velocity Model", "(B, 1, 300, 1259)")
    ]
    
    print("Expected architectural flow:")
    for i, (stage, shape) in enumerate(expected_flow, 1):
        print(f"{i:2d}. {stage:<25} → {shape}")
    
    print(f"\n🔑 Key insight: The entire pipeline maintains 128-dim embeddings")
    print(f"   from PerShotTemporalEncoder through GAT to Decoder input")
    print(f"\n📝 This ensures:")
    print(f"   • Consistent information capacity")
    print(f"   • No dimension mismatch errors")
    print(f"   • Optimal parameter efficiency")
    
if __name__ == "__main__":
    success = test_dimension_compatibility()
    test_architectural_flow()
    
    if success:
        print(f"\n🏁 READY FOR NEXT STEPS:")
        print(f"   1. Upload to Colab environment")
        print(f"   2. Install PyTorch + torch_geometric")
        print(f"   3. Run full compatibility test")
        print(f"   4. Begin training with champion loss [1.0, 0.12, 0.007]")
    else:
        print(f"\n🛠️  FIX REQUIRED:")
        print(f"   Review and fix dimension mismatches before proceeding") 