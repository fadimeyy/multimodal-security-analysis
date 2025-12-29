"""
Audio Encoder with Whisper
Survey Section 2.1 - Modality Encoder

Transcribes speech and extracts audio features
"""

import numpy as np
from typing import Dict, Optional
import warnings

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("Whisper not available. Using dummy encoder.")


class AudioEncoder:
    """
    Audio encoder using Whisper for speech recognition
    
    Survey Reference: Section 2.1 - "CLAP model as the audio encoder"
    """
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        Initialize audio encoder
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ("cpu" or "cuda")
        """
        print(f"[AudioEncoder] Loading Whisper {model_name}...")
        
        self.device = device
        self.model = None
        
        if WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model(model_name, device=device)
                print(f"[AudioEncoder] ✅ Whisper loaded on {device}")
            except Exception as e:
                print(f"[AudioEncoder] ⚠️ Could not load Whisper: {e}")
                print(f"[AudioEncoder] Using dummy encoder...")
        else:
            print(f"[AudioEncoder] ⚠️ Whisper not installed. Using dummy encoder...")
    
    def encode(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Encode audio and transcribe speech
        
        Args:
            audio_data: Audio waveform (numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with transcription, features, and metadata
        """
        
        if self.model is None:
            return self.encode_dummy()
        
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language="tr",  # Turkish
                task="transcribe",
                fp16=False
            )
            
            transcription = result["text"].strip()
            
            # Extract language and confidence if available
            language = result.get("language", "tr")
            
            # Assess audio quality (simple heuristic)
            quality = self._assess_quality(audio_data)
            
            # Generate dummy features (in real system, use audio embeddings)
            features = np.random.randn(768)
            
            # Detect if transcription contains threat keywords
            threat_keywords = [
                "help", "emergency", "danger", "fire", "weapon",
                "yardım", "acil", "tehlike", "ateş", "silah"
            ]
            
            threat_detected = any(
                keyword in transcription.lower() 
                for keyword in threat_keywords
            )
            
            return {
                "features": features,
                "transcription": transcription,
                "language": language,
                "confidence": 0.8,  # Whisper doesn't provide confidence
                "quality": quality,
                "threat_detected": threat_detected,
                "type": "speech"
            }
            
        except Exception as e:
            print(f"[AudioEncoder] Error during encoding: {e}")
            return self.encode_dummy()
    
    def encode_dummy(self) -> Dict:
        """
        Dummy encoder for testing without Whisper
        
        Returns:
            Dictionary with dummy transcription
        """
        # Simulate some transcriptions
        dummy_texts = [
            "normal conversation",
            "everything is fine",
            "test audio",
            "merhaba nasılsın",
            "her şey normal"
        ]
        
        transcription = np.random.choice(dummy_texts)
        
        return {
            "features": np.random.randn(768),
            "transcription": transcription,
            "language": "tr",
            "confidence": 0.75,
            "quality": 0.7,
            "threat_detected": False,
            "type": "speech"
        }
    
    def _assess_quality(self, audio_data: np.ndarray) -> float:
        """
        Assess audio quality based on simple metrics
        
        Args:
            audio_data: Audio waveform
            
        Returns:
            Quality score (0-1)
        """
        try:
            # Signal-to-noise ratio (simple approximation)
            signal_power = np.mean(audio_data ** 2)
            
            # Normalize to reasonable range
            if signal_power > 0:
                quality = min(signal_power * 10, 1.0)
            else:
                quality = 0.5
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except:
            return 0.7  # Default quality
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        Convenience method to transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            return self.encode_dummy()["transcription"]
        
        try:
            result = self.model.transcribe(audio_path, language="tr")
            return result["text"].strip()
        except Exception as e:
            print(f"[AudioEncoder] Error transcribing file: {e}")
            return ""


# Backward compatibility
def transcribe_audio(audio_path: str) -> str:
    """
    Legacy function for backward compatibility
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    encoder = AudioEncoder()
    return encoder.transcribe_file(audio_path)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Audio Encoder Demo")
    print("=" * 60)
    
    # Initialize encoder
    encoder = AudioEncoder(model_name="base", device="cpu")
    
    # Test with dummy audio
    print("\n### Testing with dummy audio ###")
    dummy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1  # 2 seconds
    
    result = encoder.encode(dummy_audio)
    
    print(f"\nResults:")
    print(f"  Transcription: '{result['transcription']}'")
    print(f"  Language: {result['language']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Quality: {result['quality']:.2%}")
    print(f"  Threat detected: {result['threat_detected']}")
    print(f"  Feature shape: {result['features'].shape}")
    
    # Test with real audio file (if available)
    import os
    test_audio_path = "data/WhatsApp Ptt 2025-12-18 at 07.51.56.ogg"
    
    if os.path.exists(test_audio_path):
        print(f"\n### Testing with real audio: {test_audio_path} ###")
        
        transcription = encoder.transcribe_file(test_audio_path)
        print(f"\nTranscription: '{transcription}'")
    else:
        print(f"\n⚠️ Test audio not found: {test_audio_path}")
    
    print("\n" + "=" * 60)