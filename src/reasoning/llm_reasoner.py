"""
LLM-Based Security Reasoning
Ollama (Local) + Rule-based fallback
MINIMAL VERSION - No warnings, no unused imports
"""

import os
import json
from typing import Dict, List

# Try to import Ollama (optional)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LLMReasoner:
    """Lightweight LLM reasoner: Ollama (local) + Rule-based fallback"""

    def __init__(self, provider: str = "rule-based"):
        self.provider = provider.lower()
        self.client = None
        self.model = None
        self.mode = "rule-based"

        self._init_provider()

    # ===============================
    # Provider Initialization
    # ===============================
    def _init_provider(self):
        if self.provider == "ollama":
            self._init_ollama()
        else:
            print("[LLMReasoner] ℹ️ Rule-based mode enabled")

    def _init_ollama(self):
        """Initialize Ollama (Local LLM)"""
        if not OLLAMA_AVAILABLE:
            print("[LLMReasoner] ⚠️ 'requests' library not found")
            print("[LLMReasoner] ℹ️ Using rule-based fallback")
            return
        
        try:
            # Test connection to Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                self.client = "ollama"
                self.model = "llama3"
                self.mode = "llm"
                print(f"[LLMReasoner] ✅ Ollama initialized ({self.model})")
                print("[LLMReasoner] 💡 Local inference ready")
            else:
                raise Exception(f"Ollama HTTP {response.status_code}")
                
        except Exception as e:
            print(f"[LLMReasoner] ⚠️ Ollama not available: {e}")
            print("[LLMReasoner] 💡 Start Ollama with: ollama serve")
            print("[LLMReasoner] ℹ️ Using rule-based fallback")
            self.mode = "rule-based"

    # ===============================
    # Public API
    # ===============================
    def reason(
        self,
        visual_report: Dict,
        audio_report: Dict,
        video_report: Dict,
        language: str = "tr"
    ) -> Dict:
        """Main reasoning entry point"""

        if self.mode == "llm":
            try:
                return self._llm_reasoning(
                    visual_report, audio_report, video_report, language
                )
            except Exception as e:
                print(f"[LLMReasoner] ⚠️ LLM failed → Rule-based ({e})")

        return self._rule_based_reasoning(
            visual_report, audio_report, video_report, language
        )

    # ===============================
    # LLM Reasoning
    # ===============================
    def _llm_reasoning(self, visual, audio, video, language):
        """LLM-based threat reasoning (Ollama only)"""
        prompt = self._create_prompt(visual, audio, video, language)
        result = self._call_ollama(prompt)

        result.update({
            "mode": "llm",
            "provider": "ollama",
            "model": self.model
        })

        return result

    def _call_ollama(self, prompt: str) -> Dict:
        """Call Ollama local LLM"""
        print("[LLMReasoner] 🔄 Sending request to Ollama...")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )

        print("[LLMReasoner] ✅ Ollama response received")
        
        result_json = response.json()
        return self._safe_json_parse(result_json["response"])

    # ===============================
    # Prompt Engineering
    # ===============================
    def _create_prompt(self, visual, audio, video, language):
        """Create structured prompt for LLM"""
        lang = "Turkish" if language == "tr" else "English"

        return f"""You are a security analysis AI. Analyze this multimodal data and assess threat level.

INPUT DATA:

VISUAL:
- Detected objects: {visual.get("objects", [])}
- Scene type: {visual.get("scene_type", "unknown")}
- Object count: {visual.get("object_count", 0)}

AUDIO:
- Transcription: "{audio.get("transcription", "")}"
- Audio type: {audio.get("type", "unknown")}
- Confidence: {audio.get("confidence", 0.0)}

VIDEO:
- Motion detected: {video.get("motion_detected", False)}
- Motion intensity: {video.get("motion_intensity", "none")}

THREAT ASSESSMENT RULES:
1. DANGER: Weapons (knife, gun) + Emergency keywords (help, yardım, acil)
2. SUSPICIOUS: One anomaly present (weapon OR emergency audio OR high motion)
3. SAFE: Normal objects and calm audio

RESPONSE FORMAT (strict JSON only):
{{
  "threat_level": "safe|suspicious|danger",
  "confidence": 0.85,
  "explanation": "Brief explanation in {lang}",
  "reasoning_steps": ["step1", "step2", "step3"],
  "recommendations": ["action1", "action2"]
}}

Respond with ONLY the JSON object, no markdown, no extra text."""

    # ===============================
    # Safe JSON Parse
    # ===============================
    def _safe_json_parse(self, text: str) -> Dict:
        """Safely parse JSON from LLM response"""
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON found in LLM response")

        json_str = text[start:end + 1]
        return json.loads(json_str)

    # ===============================
    # Rule-Based Fallback (ENHANCED)
    # ===============================
    def _rule_based_reasoning(self, visual, audio, video, language):
        """Enhanced rule-based reasoning with aggressive threat detection"""
        
        # Threat keywords
        weapon_keywords = [
            "knife", "knives", "blade", "gun", "pistol", "weapon",
            "bıçak", "silah", "tabanca"
        ]
        
        emergency_keywords = [
            "help", "yardım", "acil", "emergency", "imdat",
            "danger", "tehlike", "fire", "yangın"
        ]
        
        # Extract data
        visual_objects = visual.get("objects", [])
        audio_text = audio.get("transcription", "").lower()
        motion_detected = video.get("motion_detected", False)
        motion_intensity = video.get("motion_intensity", "none").lower()
        
        # Threat detection
        visual_str = ' '.join([obj.lower() for obj in visual_objects])
        
        visual_threat = any(weapon in visual_str for weapon in weapon_keywords)
        audio_threat = any(word in audio_text for word in emergency_keywords)
        motion_threat = motion_detected and motion_intensity in ['high', 'medium']
        
        # Debug output
        print("\n" + "="*60)
        print("[THREAT DETECTION DEBUG]")
        print("="*60)
        print(f"Visual objects: {visual_objects}")
        print(f"Audio text: '{audio_text}'")
        print(f"Motion: {motion_intensity}")
        print("-"*60)
        print(f"Visual threat: {visual_threat}")
        print(f"Audio threat: {audio_threat}")
        print(f"Motion threat: {motion_threat}")
        print("="*60 + "\n")
        
        # Threat level decision
        threat_count = sum([visual_threat, audio_threat, motion_threat])
        
        # AGGRESSIVE: Weapon = immediate danger
        if visual_threat:
            level, conf = "danger", 0.92
            print("⚠️ WEAPON DETECTED → DANGER")
        elif threat_count >= 2:
            level, conf = "danger", 0.85
            print("⚠️ Multiple threats → DANGER")
        elif threat_count == 1:
            level, conf = "suspicious", 0.70
            print("⚠️ Single threat → SUSPICIOUS")
        else:
            level, conf = "safe", 0.75
            print("✅ No threats → SAFE")
        
        # Generate explanation
        if language == "tr":
            if level == "danger":
                explanation = "⚠️ TEHLİKE: Tehlikeli nesne veya acil durum tespit edildi"
            elif level == "suspicious":
                explanation = "⚡ ŞÜPHELİ: Olağandışı aktivite gözlemlendi"
            else:
                explanation = "✅ GÜVENLİ: Normal aktivite"
        else:
            if level == "danger":
                explanation = "⚠️ DANGER: Weapon or emergency detected"
            elif level == "suspicious":
                explanation = "⚡ SUSPICIOUS: Unusual activity observed"
            else:
                explanation = "✅ SAFE: Normal activity"
        
        # Reasoning steps
        reasoning_steps = []
        
        if visual_threat:
            reasoning_steps.append(f"⚠️ GÖRSEL: Tehlikeli nesne tespit edildi ({visual_objects})")
        else:
            reasoning_steps.append(f"✅ Görsel: Normal nesneler")
        
        if audio_threat:
            reasoning_steps.append(f"⚠️ SES: Acil durum ifadesi: '{audio_text[:50]}'")
        else:
            reasoning_steps.append(f"✅ Ses: Normal")
        
        if motion_threat:
            reasoning_steps.append(f"⚠️ HAREKET: Yüksek yoğunluk ({motion_intensity})")
        else:
            reasoning_steps.append(f"✅ Hareket: Normal")
        
        reasoning_steps.append(f"📊 Tehdit Sayısı: {threat_count}/3")
        reasoning_steps.append(f"🎯 Karar: {level.upper()} ({conf:.0%})")
        
        return {
            "threat_level": level,
            "confidence": conf,
            "explanation": explanation,
            "reasoning_steps": reasoning_steps,
            "recommendations": self._get_recommendations(level, language),
            "mode": "rule-based",
            "model": "rule-based-v2-aggressive"
        }

    def _get_recommendations(self, level: str, language: str) -> List[str]:
        """Get threat-appropriate recommendations"""
        
        if language == "tr":
            recommendations = {
                "danger": ["🚨 Güvenliği uyar", "🔒 Alanı kapat", "📞 112'yi ara"],
                "suspicious": ["👁️ İzlemeyi artır", "🔔 Personeli bilgilendir"],
                "safe": ["✅ Rutin izlemeye devam"]
            }
        else:
            recommendations = {
                "danger": ["🚨 Alert security", "🔒 Lock down area", "📞 Call emergency"],
                "suspicious": ["👁️ Increase monitoring", "🔔 Notify staff"],
                "safe": ["✅ Continue routine monitoring"]
            }
        
        return recommendations.get(level, recommendations["safe"])


# ===============================
# QUICK TEST
# ===============================
if __name__ == "__main__":
    print("="*60)
    print("LLM Reasoner Test (Minimal Version)")
    print("="*60)
    
    # Test with rule-based
    reasoner = LLMReasoner(provider="rule-based")

    result = reasoner.reason(
        visual_report={
            "objects": ["person", "knife"],
            "scene_type": "corridor",
            "object_count": 2
        },
        audio_report={
            "transcription": "Yardım edin!",
            "confidence": 0.9,
            "type": "speech"
        },
        video_report={
            "motion_detected": True,
            "motion_intensity": "high"
        },
        language="tr"
    )

    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))