"""
🔒 Multimodal Security Analysis System - PRODUCTION VERSION
Full Pipeline: YOLO + Whisper + Video Analysis + LLM Reasoning
Master's Thesis Project - 2025
ULTRA-CLEAN VERSION - Ollama + Rule-based ONLY
"""

import streamlit as st
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
try:
    from src.reasoning.llm_reasoner import LLMReasoner
    from src.encoders.image_encoder import ImageEncoder
    from src.encoders.audio_encoder import AudioEncoder
    from src.encoders.video_encoder import VideoEncoder
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"⚠️ Module import failed: {e}")

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="🔒 Multimodal Security System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .threat-danger {
        background-color: #fee2e2;
        border-left: 5px solid #dc2626;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .threat-suspicious {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .threat-safe {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'encoders_loaded' not in st.session_state:
    st.session_state.encoders_loaded = False
    st.session_state.analysis_count = 0

# Load encoders (cache for performance)
@st.cache_resource
def load_encoders():
    """Load all encoders once"""
    print("[System] Loading encoders...")
    
    image_enc = ImageEncoder(model_name="yolov8n.pt", device="cpu")
    audio_enc = AudioEncoder(model_name="base", device="cpu")
    video_enc = VideoEncoder(frame_buffer_size=30, motion_threshold=5.0)
    
    print("[System] ✅ All encoders loaded successfully!")
    
    return image_enc, audio_enc, video_enc

# Header
st.markdown("""
<div class="main-header">
    <h1>🔒 Multimodal Security Analysis System</h1>
    <p>Context-Aware Multimodal Fusion | YOLO + Whisper + Temporal + LLM</p>
    <p style="font-size: 0.9rem;">Master's Thesis Project | Production Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # LLM Provider
    st.subheader("🧠 AI Reasoning Engine")
    provider_option = st.radio(
        "Select Provider:",
        ["📏 Rule-based (Baseline)", "💻 Ollama (Local)"],
        help="""
        **Rule-based**: Fast, no dependencies ✅
        **Ollama**: Local LLM, privacy-focused
        """
    )
    
    provider_map = {
        "📏 Rule-based (Baseline)": "rule-based",
        "💻 Ollama (Local)": "ollama"
    }
    selected_provider = provider_map[provider_option]
    
    # Provider info
    if selected_provider == "ollama":
        st.info("💻 Ollama Local\n\n✅ Privacy-focused\n⚠️ Requires local setup")
    else:
        st.success("📏 Rule-based\n\n✅ Always works\n✅ No dependencies")
    
    st.divider()
    
    # Language
    st.subheader("🌐 Language")
    language_option = st.selectbox("Select:", ["Türkçe 🇹🇷", "English 🇬🇧"])
    language = "tr" if "Türkçe" in language_option else "en"
    
    st.divider()
    
    # System metrics
    st.subheader("📊 System Status")
    st.metric("Analyses Completed", st.session_state.analysis_count)
    
    if MODULES_AVAILABLE:
        st.success("✅ All modules loaded")
    else:
        st.error("❌ Modules not available")
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About System"):
        st.markdown("""
        **Full Pipeline Architecture:**
        
        1. **ImageEncoder** → YOLOv8n (object detection)
        2. **AudioEncoder** → Whisper (speech recognition)
        3. **VideoEncoder** → Temporal analysis (motion tracking)
        4. **LLM Reasoner** → Final decision (Ollama/Rules)
        
        **Models:**
        - YOLO: yolov8n (11MB)
        - Whisper: base (142MB)
        - LLM: Ollama/Rule-based
        
        **Features:**
        - ✅ Full video support
        - ✅ Frame-by-frame analysis
        - ✅ Temporal motion tracking
        - ✅ Cross-modal fusion
        - ✅ Real-time processing
        """)

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🎬 Real-time Analysis", "📋 Test Scenarios", "ℹ️ System Info"])

# TAB 1: Real-time Analysis
with tab1:
    st.header("🔴 LIVE: Real-time Multimodal Security Analysis")
    st.markdown("**Upload image, video, or audio for full pipeline processing**")
    
    # Input columns
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.subheader("📸 Visual Input")
        
        upload_mode = st.radio(
            "Input Mode:",
            ["📁 Upload Image", "📷 Camera", "🎥 Upload Video"],
            horizontal=True,
            key="visual_mode"
        )
        
        uploaded_visual = None
        visual_type = None
        
        if upload_mode == "📁 Upload Image":
            uploaded_visual = st.file_uploader(
                "Upload security camera image",
                type=["jpg", "jpeg", "png"],
                key="img_upload"
            )
            visual_type = "image"
            
        elif upload_mode == "📷 Camera":
            uploaded_visual = st.camera_input("Take picture")
            visual_type = "image"
            
        else:  # Video
            uploaded_visual = st.file_uploader(
                "Upload security camera video",
                type=["mp4", "avi", "mov", "mkv"],
                key="vid_upload",
                help="Upload video for temporal threat analysis"
            )
            visual_type = "video"
        
        if uploaded_visual:
            if visual_type == "image":
                st.image(uploaded_visual, caption="Input Image", use_container_width=True)
            else:
                st.video(uploaded_visual)
                st.info("🎥 Video uploaded! Click ANALYZE to process frames.")
    
    with input_col2:
        st.subheader("🎤 Audio Input (Optional)")
        
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "m4a", "ogg"],
            key="audio_upload",
            help="Optional: Upload audio for speech analysis"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio)
        else:
            st.info("💡 Audio is optional. Visual-only analysis available.")
    
    # Video processing options
    if visual_type == "video" and uploaded_visual:
        st.divider()
        st.subheader("🎥 Video Processing Options")
        
        proc_col1, proc_col2 = st.columns(2)
        
        with proc_col1:
            max_frames = st.slider(
                "Maximum frames to analyze",
                min_value=10,
                max_value=100,
                value=30,
                step=10,
                help="Number of frames to process"
            )
        
        with proc_col2:
            st.metric("Estimated processing time", f"~{max_frames * 0.1:.1f}s")
    
    # Analysis section
    st.divider()
    
    if uploaded_visual:
        
        # Analysis button
        if st.button("🚀 START FULL PIPELINE ANALYSIS", type="primary", use_container_width=True, key="analyze_main"):
            
            if not MODULES_AVAILABLE:
                st.error("❌ System modules not available!")
            else:
                
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # STEP 0: Load encoders
                    status_text.text("🔄 Loading AI models...")
                    progress_bar.progress(5)
                    
                    image_encoder, audio_encoder, video_encoder = load_encoders()
                    st.session_state.encoders_loaded = True
                    
                    # STEP 1: Process Visual
                    status_text.text("👁️ Processing visual data with YOLO...")
                    progress_bar.progress(15)
                    
                    visual_result = None
                    video_summary = None
                    frame_count = 0
                    all_detected_objects = []
                    
                    if visual_type == "image":
                        # Process single image
                        image_bytes = uploaded_visual.read()
                        image = Image.open(io.BytesIO(image_bytes))
                        image_np = np.array(image)
                        
                        # Convert to BGR
                        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                        
                        visual_result = image_encoder.encode(image_np)
                        all_detected_objects = visual_result['objects']
                        
                    else:  # video
                        status_text.text("🎥 Processing video frames...")
                        
                        # Save video temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            tmp.write(uploaded_visual.read())
                            video_path = tmp.name
                        
                        cap = cv2.VideoCapture(video_path)
                        
                        if not cap.isOpened():
                            st.error("❌ Could not open video file")
                            raise Exception("Video file error")
                        
                        # Get video info
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = total_frames / fps if fps > 0 else 0
                        
                        st.info(f"📊 Video: {fps:.1f} FPS | {total_frames} frames | {duration:.1f}s")
                        
                        # Reset encoder
                        video_encoder.reset()
                        
                        # Process frames
                        frame_count = 0
                        frames_to_process = min(max_frames, total_frames)
                        
                        while frame_count < frames_to_process:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Update progress
                            video_progress = 15 + int((frame_count / frames_to_process) * 35)
                            progress_bar.progress(video_progress)
                            status_text.text(f"🎥 Frame {frame_count + 1}/{frames_to_process}...")
                            
                            # Detect objects
                            frame_visual = image_encoder.encode(frame)
                            objects = frame_visual['objects']
                            all_detected_objects.extend(objects)
                            
                            # Add to video encoder
                            video_encoder.encode(frame, objects=objects)
                            
                            frame_count += 1
                        
                        cap.release()
                        
                        # Clean up
                        try:
                            os.unlink(video_path)
                        except:
                            pass
                        
                        # Get summary
                        video_summary = video_encoder.get_temporal_summary()
                        
                        # Aggregated results
                        unique_objects = list(set(all_detected_objects))
                        
                        visual_result = {
                            'objects': unique_objects,
                            'num_objects': len(unique_objects),
                            'confidence': 0.85,
                            'quality': 0.80,
                            'confidences': [0.85] * len(unique_objects)
                        }
                        
                        st.success(f"✅ Video: {frame_count} frames | {len(unique_objects)} unique objects")
                    
                    if visual_type == "image":
                        st.success(f"✅ Visual: {visual_result['num_objects']} objects detected")
                    
                    # STEP 2: Process Audio
                    status_text.text("🎤 Processing audio with Whisper...")
                    progress_bar.progress(55)
                    
                    audio_result = None
                    
                    if uploaded_audio:
                        # Save audio temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(uploaded_audio.read())
                            audio_path = tmp.name
                        
                        transcription = audio_encoder.transcribe_file(audio_path)
                        
                        # Clean up
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
                        
                        audio_result = {
                            "transcription": transcription,
                            "type": "speech",
                            "confidence": 0.85,
                            "language": language,
                            "duration": 5.0
                        }
                        
                        st.success(f"✅ Audio: \"{transcription[:50]}{'...' if len(transcription) > 50 else ''}\"")
                    else:
                        audio_result = {
                            "transcription": "",
                            "type": "silence",
                            "confidence": 0.0,
                            "language": language,
                            "duration": 0.0
                        }
                        st.info("ℹ️ No audio - visual-only analysis")
                    
                    # STEP 3: Temporal Analysis
                    status_text.text("🎥 Analyzing temporal patterns...")
                    progress_bar.progress(70)
                    
                    if visual_type == "video" and video_summary:
                        video_result = {
                            "motion_detected": video_summary.get('motion_pattern', 'static') != 'static',
                            "motion_intensity": video_summary.get('motion_pattern', 'medium'),
                            "duration": frame_count * 0.033,
                            "frames_processed": frame_count,
                            "avg_motion": video_summary.get('avg_motion', 0.0),
                            "object_stability": video_summary.get('object_stability', 0.0)
                        }
                        
                        st.success(f"✅ Temporal: {video_summary.get('motion_pattern', 'unknown').upper()} motion")
                    else:
                        video_result = {
                            "motion_detected": False,
                            "motion_intensity": "none",
                            "duration": 0.0,
                            "frames_processed": 1,
                            "avg_motion": 0.0,
                            "object_stability": 1.0
                        }
                    
                    # STEP 4: LLM Reasoning
                    status_text.text("🧠 Running LLM reasoning...")
                    progress_bar.progress(85)
                    
                    llm_visual = {
                        "objects": visual_result['objects'],
                        "scene_type": "video_sequence" if visual_type == "video" else "single_frame",
                        "object_count": visual_result['num_objects']
                    }
                    
                    reasoner = LLMReasoner(provider=selected_provider)
                    
                    start_time = time.time()
                    final_result = reasoner.reason(
                        llm_visual,
                        audio_result,
                        video_result,
                        language=language
                    )
                    elapsed = time.time() - start_time
                    
                    # Complete
                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.session_state.analysis_count += 1
                    
                    # DISPLAY RESULTS
                    st.divider()
                    st.subheader("🎯 MULTIMODAL ANALYSIS RESULTS")
                    
                    # Threat card
                    threat_level = final_result['threat_level']
                    
                    if threat_level == "danger":
                        card_class = "threat-danger"
                        emoji = "🚨"
                    elif threat_level == "suspicious":
                        card_class = "threat-suspicious"
                        emoji = "⚠️"
                    else:
                        card_class = "threat-safe"
                        emoji = "✅"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h2>{emoji} THREAT LEVEL: {threat_level.upper()}</h2>
                        <h3>{final_result['explanation']}</h3>
                        <p><strong>Confidence:</strong> {final_result['confidence']:.0%}</p>
                        <p><strong>AI Model:</strong> {final_result.get('model', 'N/A')} ({final_result.get('mode', 'N/A')})</p>
                        <p><strong>Processing Time:</strong> {elapsed:.3f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    st.divider()
                    st.subheader("📊 Modality-wise Analysis")
                    
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown("### 👁️ Visual Analysis")
                        st.write(f"**Model:** YOLOv8n")
                        if visual_type == "video":
                            st.write(f"**Type:** Video ({frame_count} frames)")
                        else:
                            st.write(f"**Type:** Single Image")
                        st.write(f"**Objects:** {', '.join(visual_result['objects'][:5])}")
                        if len(visual_result['objects']) > 5:
                            st.write(f"  _(+{len(visual_result['objects']) - 5} more)_")
                        st.write(f"**Count:** {visual_result['num_objects']}")
                    
                    with detail_col2:
                        st.markdown("### 🔊 Audio Analysis")
                        st.write(f"**Model:** Whisper Base")
                        if audio_result['transcription']:
                            st.write(f"**Text:** {audio_result['transcription'][:50]}...")
                        else:
                            st.write(f"**Text:** (no audio)")
                        st.write(f"**Type:** {audio_result['type']}")
                        st.write(f"**Language:** {audio_result['language'].upper()}")
                    
                    with detail_col3:
                        st.markdown("### 🎥 Temporal Analysis")
                        st.write(f"**Type:** {visual_type.capitalize()}")
                        st.write(f"**Motion:** {'Yes' if video_result['motion_detected'] else 'No'}")
                        st.write(f"**Intensity:** {video_result['motion_intensity'].upper()}")
                        if visual_type == "video":
                            st.write(f"**Frames:** {video_result['frames_processed']}")
                    
                    # Reasoning
                    st.divider()
                    
                    reason_col1, reason_col2 = st.columns(2)
                    
                    with reason_col1:
                        st.subheader("🔍 Reasoning Chain")
                        if final_result.get('reasoning_steps'):
                            for i, step in enumerate(final_result.get('reasoning_steps', []), 1):
                                st.write(f"**{i}.** {step}")
                        else:
                            st.write("_No detailed reasoning steps_")
                    
                    with reason_col2:
                        st.subheader("💡 Recommended Actions")
                        if final_result.get('recommendations'):
                            for rec in final_result.get('recommendations', []):
                                st.info(rec)
                        else:
                            st.write("_No specific recommendations_")
                    
                    # Export
                    st.divider()
                    
                    import json
                    export_data = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'input_type': visual_type,
                        'provider': selected_provider,
                        'visual_analysis': {
                            'objects': visual_result['objects'],
                            'count': visual_result['num_objects']
                        },
                        'audio_analysis': {
                            'transcription': audio_result['transcription'],
                            'type': audio_result['type']
                        },
                        'temporal_analysis': video_result,
                        'final_result': final_result,
                        'metrics': {
                            'processing_time': elapsed,
                            'language': language
                        }
                    }
                    
                    st.download_button(
                        "📥 Download Analysis Report (JSON)",
                        data=json.dumps(export_data, indent=2, ensure_ascii=False),
                        file_name=f"security_analysis_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    import traceback
                    with st.expander("🐛 Error Details"):
                        st.code(traceback.format_exc())

# TAB 2: Test Scenarios
with tab2:
    st.header("📋 Pre-configured Test Scenarios")
    st.info("Quick testing with sample multimodal data")
    
    st.markdown("""
    **Available test scenarios:**
    
    🟢 **Safe Scenario**
    - Office environment
    - Normal objects
    - Expected: SAFE
    
    🟡 **Suspicious Scenario**
    - Unknown person
    - Unusual motion
    - Expected: SUSPICIOUS
    
    🔴 **Danger Scenario**
    - Weapon detected
    - Emergency audio
    - Expected: DANGER
    """)

# TAB 3: System Info
with tab3:
    st.header("ℹ️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Pipeline")
        st.code("""
        Input (Image/Video + Audio)
              ↓
        ┌─────────────────────┐
        │ 1. ImageEncoder     │ → YOLOv8n
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ 2. AudioEncoder     │ → Whisper
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ 3. VideoEncoder     │ → Temporal
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ 4. LLM Reasoner     │ → Ollama/Rules
        └─────────────────────┘
              ↓
        Final Assessment
        """)
    
    with col2:
        st.subheader("📊 Models")
        st.markdown("""
        **Computer Vision:**
        - YOLOv8n (11MB)
        - Speed: ~10ms/frame
        
        **Speech Recognition:**
        - Whisper Base (142MB)
        - Languages: 99+
        
        **LLM Reasoning:**
        - Ollama (Llama 3.2)
        - Rule-based fallback
        
        **System:**
        - RAM: 4GB minimum
        - GPU: Optional
        - Python: 3.9+
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🔒 Multimodal Security Analysis System v3.0</strong></p>
    <p>Master's Thesis 2025 | Clean Production Build</p>
    <p style='font-size: 0.8rem;'>⚡ Real AI Models | 🎥 Full Video | 🛡️ Intelligent Analysis</p>
</div>
""", unsafe_allow_html=True)