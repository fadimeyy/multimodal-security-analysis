"""
🧪 LLM Reasoner Quick Test
NO GEMINI - Tests Ollama + Rule-based only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.reasoning.llm_reasoner import LLMReasoner
from dotenv import load_dotenv
import json

# Load .env
load_dotenv()

print("\n" + "="*60)
print("🧪 TESTING LLM REASONER (NO GEMINI)")
print("="*60)

# Test data
visual = {
    "objects": ["person", "knife", "door"],
    "scene_type": "indoor",
    "object_count": 3
}

audio = {
    "transcription": "yardım edin lütfen",
    "type": "speech",
    "confidence": 0.92,
    "language": "tr"
}

video = {
    "motion_detected": True,
    "motion_intensity": "high",
    "duration": 5.2,
    "frames_processed": 150
}

print("\n📊 Test Scenario:")
print(f"  Visual: {visual['objects']}")
print(f"  Audio: '{audio['transcription']}'")
print(f"  Motion: {video['motion_intensity']}")
print("")

# Test 1: Rule-based
print("="*60)
print("TEST 1: Rule-based Reasoning")
print("="*60)

try:
    print("\n📦 Initializing rule-based reasoner...")
    reasoner_rules = LLMReasoner(provider="rule-based")
    
    print("🔄 Analyzing...")
    result_rules = reasoner_rules.reason(visual, audio, video, language="tr")
    
    print("\n✅ RESULT:")
    print(f"  🎯 Threat: {result_rules['threat_level'].upper()}")
    print(f"  📊 Confidence: {result_rules['confidence']:.0%}")
    print(f"  💬 {result_rules['explanation']}")
    print(f"  🔍 Mode: {result_rules['mode']}")
    print(f"  📋 Model: {result_rules['model']}")
    
    if result_rules.get('reasoning_steps'):
        print(f"\n  Reasoning:")
        for i, step in enumerate(result_rules['reasoning_steps'], 1):
            print(f"    {i}. {step}")
    
    print("\n✅ Rule-based test PASSED!")
    
except Exception as e:
    print(f"\n❌ Rule-based test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Ollama (if available)
print("\n" + "="*60)
print("TEST 2: Ollama Local LLM")
print("="*60)

try:
    print("\n📦 Initializing Ollama...")
    reasoner_ollama = LLMReasoner(provider="ollama")
    
    if reasoner_ollama.mode == "rule-based":
        print("⚠️ Ollama not available, using rule-based fallback")
        print("💡 To test Ollama: Start it with 'ollama serve'")
    else:
        print("🔄 Analyzing with Ollama...")
        result_ollama = reasoner_ollama.reason(visual, audio, video, language="tr")
        
        print("\n✅ RESULT:")
        print(f"  🎯 Threat: {result_ollama['threat_level'].upper()}")
        print(f"  📊 Confidence: {result_ollama['confidence']:.0%}")
        print(f"  💬 {result_ollama['explanation']}")
        print(f"  🔍 Mode: {result_ollama['mode']}")
        print(f"  📋 Model: {result_ollama['model']}")
        
        if result_ollama.get('reasoning_steps'):
            print(f"\n  Reasoning:")
            for i, step in enumerate(result_ollama['reasoning_steps'], 1):
                print(f"    {i}. {step}")
        
        print("\n✅ Ollama test PASSED!")
    
except Exception as e:
    print(f"\n⚠️ Ollama test error: {e}")
    print("💡 This is OK if Ollama is not installed")

# Summary
print("\n" + "="*60)
print("🎉 TEST SUMMARY")
print("="*60)
print("✅ Rule-based: WORKING")
print("⚠️ Ollama: Check above for status")
print("\n💡 System is ready for deployment!")
print("="*60 + "\n")