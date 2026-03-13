#!/usr/bin/env python3
"""
Test script to verify remix functionality works correctly.
This script tests the remix functionality with a simple audio file.
"""

import os
import sys
import numpy as np
import soundfile as sf
from audio_utils.remix import handle_remix
from llm_backend.session_manager import save_file_to_db
import tempfile

def create_test_audio(duration_seconds=5, sample_rate=44100):
    """Create a simple test audio file with different frequency content for each 'stem'"""
    
    # Create different frequency content to simulate different stems
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Simulate vocals (higher frequency)
    vocals = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    # Simulate drums (percussive, lower frequency)
    drums = 0.4 * np.sin(2 * np.pi * 80 * t) * np.exp(-t * 2)  # Decaying low freq
    
    # Simulate bass (very low frequency)
    bass = 0.5 * np.sin(2 * np.pi * 60 * t)  # Low bass note
    
    # Simulate other (mid frequency)
    other = 0.2 * np.sin(2 * np.pi * 220 * t)  # Lower A note
    
    # Mix them together for the original audio
    mixed = vocals + drums + bass + other
    
    # Make stereo
    stereo_audio = np.column_stack([mixed, mixed])
    
    # Normalize to prevent clipping
    stereo_audio = stereo_audio / np.max(np.abs(stereo_audio)) * 0.8
    
    return stereo_audio, sample_rate

def test_remix_volume_changes():
    """Test that volume changes are actually applied"""
    print("=== Testing Remix Volume Changes ===")
    
    # Create test audio
    audio_data, sr = create_test_audio(duration_seconds=3)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_data, sr)
        test_audio_path = temp_file.name
    
    try:
        # Mock session setup
        session_id = "test_session_123"
        
        # Mock the database function to return our test file
        original_get_file = None
        try:
            from llm_backend.session_manager import get_file_from_db
            original_get_file = get_file_from_db
        except:
            pass
            
        # Monkey patch the get_file_from_db function
        import llm_backend.session_manager
        llm_backend.session_manager.get_file_from_db = lambda session_id, file_type="uploaded": test_audio_path
        
        # Test 1: Make vocals much louder
        print("\nTest 1: Making vocals much louder...")
        intent1 = {
            "type": "remix",
            "instructions": {
                "volumes": {
                    "vocals": 2.0,  # Much louder
                    "drums": 1.0,
                    "bass": 1.0,
                    "other": 1.0
                }
            }
        }
        
        result1 = handle_remix(intent1, session_id)
        print(f"Result 1: {result1}")
        
        # Test 2: Solo vocals (mute everything else)
        print("\nTest 2: Solo vocals...")
        intent2 = {
            "type": "remix", 
            "instructions": {
                "volumes": {
                    "vocals": 1.0,
                    "drums": 0.0,   # Muted
                    "bass": 0.0,    # Muted
                    "other": 0.0    # Muted
                }
            }
        }
        
        result2 = handle_remix(intent2, session_id)
        print(f"Result 2: {result2}")
        
        # Test 3: Add reverb to vocals
        print("\nTest 3: Add reverb to vocals...")
        intent3 = {
            "type": "remix",
            "instructions": {
                "volumes": {
                    "vocals": 1.0,
                    "drums": 1.0,
                    "bass": 1.0,
                    "other": 1.0
                },
                "reverb": {
                    "vocals": 0.8  # Heavy reverb
                }
            }
        }
        
        result3 = handle_remix(intent3, session_id)
        print(f"Result 3: {result3}")
        
        # Test 4: Pitch shift vocals
        print("\nTest 4: Pitch shift vocals up...")
        intent4 = {
            "type": "remix",
            "instructions": {
                "volumes": {
                    "vocals": 1.0,
                    "drums": 1.0,
                    "bass": 1.0,
                    "other": 1.0
                },
                "pitch_shift": {
                    "vocals": 4  # Up 4 semitones
                }
            }
        }
        
        result4 = handle_remix(intent4, session_id)
        print(f"Result 4: {result4}")
        
        print("\n=== All tests completed! ===")
        print("Check the 'separated' folder for the generated remix files.")
        print("You should hear clear differences between the original and remixed versions.")
        
        # Restore original function if it existed
        if original_get_file:
            llm_backend.session_manager.get_file_from_db = original_get_file
            
    finally:
        # Clean up temp file
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)

if __name__ == "__main__":
    # Make sure separated directory exists
    os.makedirs("separated", exist_ok=True)
    
    test_remix_volume_changes()
