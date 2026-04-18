import os
import threading
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from espeakng_loader import get_data_path, get_library_path, make_library_available
from kokoro import KPipeline


ROOT_DIR = Path(r"C:\StudioVoice")
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
CPU_THREADS = max(1, min(4, ((os.cpu_count() or 4) // 2) or 1))
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

VOICE_PROFILES = {
    "af_heart": "American female. Warm, intimate, and soft. Great first pick for heartfelt narration and polished explainers.",
    "af_alloy": "American female. Neutral and clean with a modern studio feel. Good all-purpose commercial voice.",
    "af_aoede": "American female. Airy and lyrical. Nice for elegant reads and gentle cinematic copy.",
    "af_bella": "American female. Bright and friendly with a clear smile in the tone. Good for social and brand content.",
    "af_jessica": "American female. Smooth and confident. Useful for premium presentation and tutorial reads.",
    "af_kore": "American female. Slightly firmer and more focused. Good for crisp instructional voiceover.",
    "af_nicole": "American female. Balanced and polished. Safe choice for general narration.",
    "af_nova": "American female. Contemporary and energetic. Useful for upbeat promo material.",
    "af_river": "American female. Relaxed and modern with a softer edge. Good for calm storytelling.",
    "am_adam": "American male. Natural, grounded, and straightforward. Good for general narration.",
    "am_echo": "American male. Slightly cooler modern tone. Nice for tech-style demos and sleek promos.",
    "am_eric": "American male. Clean and conversational. Good for courses and explainers.",
    "am_fenrir": "American male. Darker, weightier tone. Best for dramatic lines and cinematic trailers.",
    "am_liam": "American male. Friendly and relaxed. Good for approachable everyday reads.",
    "am_michael": "American male. Rich, centered, and broadcaster-like. One of the strongest studio-style male voices.",
    "am_onyx": "American male. Deep and polished with a premium edge. Good for luxury or serious material.",
    "am_puck": "American male. Light and lively. Better for playful, youthful reads.",
    "am_santa": "American male. Distinct character voice with a festive personality.",
    "bf_alice": "British female. Soft RP-style tone with a graceful delivery. Good for refined narration.",
    "bf_emma": "British female. Clear and friendly. Great for elegant but accessible voiceover.",
    "bf_isabella": "British female. Smooth and slightly premium. Useful for branded reads.",
    "bf_lily": "British female. Light, bright, and calm. Good for gentle storytelling.",
    "bm_daniel": "British male. Clear and composed. Good for documentary or instructional material.",
    "bm_fable": "British male. Story-driven and a bit theatrical. Nice for narrative pieces.",
    "bm_george": "British male. Balanced and reassuring. Good general-purpose UK narration.",
    "bm_lewis": "British male. Younger and more casual. Good for modern web content.",
}

VOICE_OPTIONS = list(VOICE_PROFILES.keys())
PIPELINES = {}
PIPELINE_LOCKS = {}
WARMUP_LOCK = threading.Lock()
WARMUP_STARTED = False
WARMUP_DONE = False
WARMUP_ERROR = ""

try:
    make_library_available()
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", get_data_path())
except Exception as exc:
    print(f"Warning: eSpeak initialization issue: {exc}")

try:
    import resemble_enhance  # type: ignore

    RESEMBLE_AVAILABLE = True
    RESEMBLE_IMPORT_ERROR = ""
except Exception as exc:
    RESEMBLE_AVAILABLE = False
    RESEMBLE_IMPORT_ERROR = str(exc)


CUSTOM_CSS = """
:root {
  --bg: #030303;
  --bg-soft: #090909;
  --panel: rgba(13, 13, 14, 0.9);
  --panel-soft: rgba(20, 20, 22, 0.74);
  --line: rgba(255, 255, 255, 0.08);
  --line-soft: rgba(255, 255, 255, 0.05);
  --text: #f3f2ee;
  --muted: #a3a3aa;
  --muted-soft: #707078;
  --accent: #ff6a1a;
  --shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
  --radius-xl: 30px;
  --radius-lg: 24px;
  --radius-md: 18px;
  --display-font: "Segoe UI Variable Display", "Bahnschrift", "Segoe UI", sans-serif;
  --body-font: "Segoe UI Variable Text", "Segoe UI", sans-serif;
}
html,
body {
  background: var(--bg);
}
body,
.gradio-container {
  min-height: 100vh;
  color: var(--text);
  font-family: var(--body-font);
  background:
    radial-gradient(60rem 24rem at 50% -6%, rgba(255, 255, 255, 0.18), transparent 42%),
    radial-gradient(24rem 18rem at 14% 18%, rgba(255, 255, 255, 0.07), transparent 50%),
    radial-gradient(18rem 16rem at 86% 34%, rgba(255, 255, 255, 0.04), transparent 50%),
    radial-gradient(26rem 14rem at 50% 100%, rgba(255, 106, 26, 0.08), transparent 42%),
    linear-gradient(180deg, #090909 0%, #040404 52%, #020202 100%);
}
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto;
  padding: 28px 18px 54px !important;
}
.app-shell {
  position: relative;
}
.app-shell::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(125deg, transparent 24%, rgba(255, 255, 255, 0.045) 42%, transparent 58%),
    radial-gradient(circle, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
  background-size: auto, 12px 12px;
  opacity: 0.22;
}
.chrome-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 22px;
  padding: 14px 18px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
  backdrop-filter: blur(14px);
}
.brand-lockup {
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: var(--display-font);
  font-size: 15px;
  letter-spacing: 0.04em;
}
.brand-mark {
  width: 11px;
  height: 11px;
  border-radius: 999px;
  background: radial-gradient(circle at 35% 35%, #ffb37a, var(--accent) 58%, #7a2400 100%);
  box-shadow: 0 0 18px rgba(255, 106, 26, 0.55);
}
.chrome-links {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.chrome-links span {
  padding: 7px 12px;
  border: 1px solid var(--line-soft);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.hero-panel {
  position: relative;
  overflow: hidden;
  margin-bottom: 22px;
  padding: 38px 34px;
  border: 1px solid var(--line);
  border-radius: var(--radius-xl);
  background:
    radial-gradient(34rem 18rem at 50% 0%, rgba(255, 255, 255, 0.15), transparent 50%),
    linear-gradient(180deg, rgba(18, 18, 19, 0.94), rgba(8, 8, 9, 0.98));
  box-shadow: var(--shadow);
}
.hero-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(124deg, transparent 32%, rgba(255, 255, 255, 0.06) 48%, transparent 60%),
    radial-gradient(circle, rgba(255, 255, 255, 0.08) 1px, transparent 1px);
  background-size: auto, 10px 10px;
  background-position: center, center;
  opacity: 0.22;
  pointer-events: none;
}
.hero-grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(260px, 0.75fr);
  gap: 24px;
  align-items: end;
}
.hero-kicker {
  margin-bottom: 14px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.hero-title {
  max-width: 760px;
  margin: 0;
  font-family: var(--display-font);
  font-size: 52px;
  line-height: 0.96;
  letter-spacing: -0.04em;
}
.hero-copy {
  max-width: 620px;
  margin: 18px 0 0;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.8;
}
.hero-side {
  justify-self: end;
  width: 100%;
  max-width: 320px;
  padding: 18px;
  border: 1px solid var(--line-soft);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.hero-side-label {
  color: var(--muted-soft);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.hero-side-value {
  margin-top: 10px;
  font-family: var(--display-font);
  font-size: 24px;
  line-height: 1.1;
}
.hero-side-copy {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.65;
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 26px;
}
.metric-card {
  position: relative;
  overflow: hidden;
  padding: 16px 18px;
  border: 1px solid var(--line-soft);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
}
.metric-card::after {
  content: "";
  position: absolute;
  inset: auto -22px -30px auto;
  width: 110px;
  height: 110px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.08), transparent 68%);
}
.metric-label {
  position: relative;
  z-index: 1;
  color: var(--muted-soft);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.metric-value {
  position: relative;
  z-index: 1;
  margin-top: 8px;
  font-family: var(--display-font);
  font-size: 18px;
  letter-spacing: -0.02em;
}
#workbench-grid {
  gap: 18px;
}
#left-column,
#right-column {
  gap: 18px;
}
#script-panel,
#audio-panel,
#status-panel,
#control-panel,
#library-panel,
#examples-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line) !important;
  border-radius: var(--radius-xl) !important;
  background: linear-gradient(180deg, rgba(18, 18, 19, 0.94), rgba(9, 9, 10, 0.96)) !important;
  box-shadow: var(--shadow) !important;
}
#script-panel::before,
#audio-panel::before,
#status-panel::before,
#control-panel::before,
#library-panel::before,
#examples-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(124deg, transparent 28%, rgba(255, 255, 255, 0.04) 46%, transparent 56%),
    radial-gradient(circle, rgba(255, 255, 255, 0.055) 1px, transparent 1px);
  background-size: auto, 12px 12px;
  opacity: 0.18;
  pointer-events: none;
}
.section-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.section-head span {
  color: var(--muted-soft);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.section-head strong {
  font-family: var(--display-font);
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.03em;
}
#script-panel,
#audio-panel,
#status-panel,
#control-panel,
#examples-panel {
  padding: 22px !important;
}
#script-input textarea,
#status-output textarea {
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  background: rgba(255, 255, 255, 0.03) !important;
  color: var(--text) !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
  backdrop-filter: blur(10px);
}
#script-input textarea {
  min-height: 320px !important;
}
#status-output textarea {
  min-height: 214px !important;
}
#script-input textarea::placeholder,
#status-output textarea::placeholder {
  color: #7a7a82 !important;
}
#control-panel input,
#control-panel textarea,
#control-panel button,
#control-panel .wrap,
#control-panel .inner,
#control-panel .container,
#control-panel select,
#script-panel input,
#script-panel textarea,
#status-panel textarea {
  color: var(--text) !important;
}
#control-panel .wrap,
#control-panel .container,
#control-panel .inner,
#control-panel input,
#control-panel textarea,
#control-panel button,
#control-panel select {
  background: rgba(255, 255, 255, 0.03) !important;
  border-color: var(--line) !important;
  border-radius: var(--radius-md) !important;
}
#control-panel .info {
  color: var(--muted-soft) !important;
}
#generate-btn {
  min-height: 56px !important;
  margin-top: 16px;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 999px !important;
  background: linear-gradient(180deg, rgba(255, 140, 72, 0.92), rgba(255, 106, 26, 0.92)) !important;
  color: #ffffff !important;
  box-shadow: 0 16px 34px rgba(255, 106, 26, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.32) !important;
  font-family: var(--display-font) !important;
  letter-spacing: 0.02em;
}
#generate-btn:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}
#audio-panel audio {
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.06);
}
.profile-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius-xl);
  padding: 22px;
  background: radial-gradient(16rem 8rem at 50% 0%, rgba(255, 255, 255, 0.12), transparent 70%), linear-gradient(180deg, rgba(16, 16, 18, 0.96), rgba(8, 8, 9, 0.98));
  box-shadow: var(--shadow);
}
.profile-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(124deg, transparent 30%, rgba(255, 255, 255, 0.05) 46%, transparent 58%),
    radial-gradient(circle, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
  background-size: auto, 12px 12px;
  opacity: 0.2;
  pointer-events: none;
}
.profile-top {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}
.profile-label {
  color: var(--muted-soft);
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.profile-title {
  margin: 8px 0 0;
  font-family: var(--display-font);
  font-size: 24px;
  letter-spacing: -0.03em;
}
.profile-code {
  padding: 9px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}
.profile-tags {
  position: relative;
  z-index: 1;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 16px;
}
.profile-tags span {
  padding: 7px 11px;
  border: 1px solid var(--line-soft);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.profile-copy {
  position: relative;
  z-index: 1;
  margin-top: 16px;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.8;
}
.profile-accent {
  position: relative;
  z-index: 1;
  margin-top: 18px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.18), transparent);
}
.library-shell {
  padding: 22px;
}
.family-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.family-card {
  position: relative;
  overflow: hidden;
  min-height: 166px;
  padding: 18px;
  border: 1px solid var(--line-soft);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.02));
}
.family-card::before {
  content: "";
  position: absolute;
  inset: auto -18px -28px auto;
  width: 96px;
  height: 96px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.09), transparent 70%);
}
.family-code {
  position: relative;
  z-index: 1;
  color: var(--accent);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.family-card h4 {
  position: relative;
  z-index: 1;
  margin: 10px 0 8px;
  font-family: var(--display-font);
  font-size: 20px;
  letter-spacing: -0.03em;
}
.family-card p {
  position: relative;
  z-index: 1;
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.7;
}
.family-list {
  position: relative;
  z-index: 1;
  margin-top: 14px;
  color: #d7d7db;
  font-size: 12px;
  line-height: 1.7;
}
.footnote {
  margin-top: 18px;
  color: var(--muted-soft);
  font-size: 12px;
  line-height: 1.8;
}
#examples-panel table,
#examples-panel tbody,
#examples-panel tr,
#examples-panel td,
#examples-panel button,
#examples-panel .gr-samples,
#examples-panel .gr-samples-table {
  background: transparent !important;
  color: var(--text) !important;
  border-color: var(--line-soft) !important;
}
@media (max-width: 980px) {
  .hero-grid {
    grid-template-columns: 1fr;
  }
  .hero-side {
    justify-self: start;
    max-width: none;
  }
  .metric-row,
  .family-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 768px) {
  .chrome-bar {
    flex-direction: column;
    align-items: flex-start;
    border-radius: 24px;
  }
  .hero-panel {
    padding: 28px 22px;
  }
  .hero-title {
    font-size: 38px;
  }
}
"""


def language_for_voice(voice_name: str) -> str:
    return "b" if voice_name.startswith("b") else "a"


def get_pipeline(voice_name: str) -> KPipeline:
    lang_code = language_for_voice(voice_name)
    if lang_code not in PIPELINES:
        PIPELINES[lang_code] = KPipeline(lang_code=lang_code, device="cpu")
    return PIPELINES[lang_code]


def get_pipeline_lock(voice_name: str) -> threading.Lock:
    lang_code = language_for_voice(voice_name)
    if lang_code not in PIPELINE_LOCKS:
        PIPELINE_LOCKS[lang_code] = threading.Lock()
    return PIPELINE_LOCKS[lang_code]


def warm_up_default_voice() -> None:
    global WARMUP_STARTED, WARMUP_DONE, WARMUP_ERROR
    if WARMUP_DONE or WARMUP_STARTED:
        return
    with WARMUP_LOCK:
        if WARMUP_DONE or WARMUP_STARTED:
            return
        WARMUP_STARTED = True
    try:
        pipeline = get_pipeline(DEFAULT_VOICE)
        pipeline_lock = get_pipeline_lock(DEFAULT_VOICE)
        with pipeline_lock:
            pipeline.load_voice(DEFAULT_VOICE)
            with torch.inference_mode():
                for result in pipeline("Warm up.", voice=DEFAULT_VOICE, speed=1.0, split_pattern=None):
                    _ = result.audio
                    break
        WARMUP_DONE = True
    except Exception as exc:
        WARMUP_ERROR = str(exc)


def start_background_warmup() -> None:
    thread = threading.Thread(target=warm_up_default_voice, name="studio-voice-warmup", daemon=True)
    thread.start()


def voice_profile(voice_name: str) -> str:
    label_map = {"a": "American", "b": "British"}
    accent = label_map.get(voice_name[0], "English")
    gender = "Female" if voice_name[1] == "f" else "Male"
    family_code = voice_name[:2].upper()
    body = VOICE_PROFILES.get(voice_name, "A pretrained Kokoro voice.")
    return f"""
    <div class="profile-panel">
      <div class="profile-top">
        <div>
          <div class="profile-label">Selected voice</div>
          <div class="profile-title">{voice_name}</div>
        </div>
        <div class="profile-code">{family_code}</div>
      </div>
      <div class="profile-tags">
        <span>{accent}</span>
        <span>{gender}</span>
        <span>Pretrained</span>
      </div>
      <div class="profile-copy">{body}</div>
      <div class="profile-accent"></div>
    </div>
    """


def voice_library_html() -> str:
    return """
    <div class="library-shell">
      <div class="section-head">
        <span>Voice families</span>
        <strong>Library</strong>
      </div>
      <div class="family-grid">
        <div class="family-card">
          <div class="family-code">AF</div>
          <h4>American Female</h4>
          <p>Warm, clean, commercial-ready voices with the smoothest range for polished narration and branded reads.</p>
          <div class="family-list">Best starts: af_heart, af_alloy, af_bella, af_nova</div>
        </div>
        <div class="family-card">
          <div class="family-code">AM</div>
          <h4>American Male</h4>
          <p>Grounded, modern, and strong on explainers, premium ads, documentary-style delivery, and dramatic trailers.</p>
          <div class="family-list">Best starts: am_michael, am_onyx, am_adam, am_echo</div>
        </div>
        <div class="family-card">
          <div class="family-code">BF</div>
          <h4>British Female</h4>
          <p>Soft, refined, and elegant voices suited to premium storytelling, calm instruction, and a more editorial tone.</p>
          <div class="family-list">Best starts: bf_emma, bf_alice, bf_isabella</div>
        </div>
        <div class="family-card">
          <div class="family-code">BM</div>
          <h4>British Male</h4>
          <p>Balanced UK voices with a sharper narrative presence, especially strong for story-led and documentary reads.</p>
          <div class="family-list">Best starts: bm_fable, bm_george, bm_daniel</div>
        </div>
      </div>
      <div class="footnote">
        Code system: <strong>a</strong> or <strong>b</strong> sets the accent family, then <strong>f</strong> or <strong>m</strong> marks female or male.
      </div>
    </div>
    """


def maybe_enhance(audio: np.ndarray, sample_rate: int, enabled: bool) -> tuple[np.ndarray, str]:
    if not enabled:
        return audio, "Enhance disabled."
    if not RESEMBLE_AVAILABLE:
        return audio, f"Resemble Enhance unavailable on this Windows setup: {RESEMBLE_IMPORT_ERROR}"
    return audio, "Resemble Enhance is installed, but its automatic Windows post-processing path still needs a custom integration layer."


def synthesize(text: str, voice: str, speed: float, enhance: bool, progress=gr.Progress(track_tqdm=False)):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text first.")

    status = [f"Starting CPU render for: {voice}"]
    yield None, "\n".join(status), voice_profile(voice)

    progress(0.05, desc="Preparing voice engine")
    pipeline = get_pipeline(voice)
    pipeline_lock = get_pipeline_lock(voice)

    chunks = []
    status.append("Waiting for voice engine...")
    yield None, "\n".join(status), voice_profile(voice)
    progress(0.18, desc="Waiting for voice engine")
    
    with pipeline_lock:
        status[-1] = "Loading selected voice..."
        yield None, "\n".join(status), voice_profile(voice)
        progress(0.32, desc="Loading selected voice")
        pipeline.load_voice(voice)
        
        status[-1] = "Synthesizing audio..."
        yield None, "\n".join(status), voice_profile(voice)
        progress(0.52, desc="Synthesizing audio")
        
        with torch.inference_mode():
            for result in pipeline(text, voice=voice, speed=float(speed), split_pattern=r"\n+"):
                if result.audio is None:
                    continue
                audio = result.audio.detach().cpu().numpy()
                chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                # Add a yield inside the heavy loop to keep connection alive on very long texts
                yield None, "\n".join(status) + f"\n... processing chunk {len(chunks)} ...", voice_profile(voice)

    if not chunks:
        raise gr.Error("Kokoro did not return audio for that text.")

    audio = np.concatenate(chunks)
    status[-1] = "Applying post-processing..."
    yield None, "\n".join(status), voice_profile(voice)
    progress(0.82, desc="Applying post-processing")
    audio, enhance_message = maybe_enhance(audio, SAMPLE_RATE, enhance)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"studio_voice_{voice}_{timestamp}.wav"
    
    status[-1] = "Saving WAV..."
    yield None, "\n".join(status), voice_profile(voice)
    progress(0.94, desc="Saving WAV")
    sf.write(output_path, audio, SAMPLE_RATE)

    warm_status = "Warmup ready." if WARMUP_DONE else "Warmup in progress." if not WARMUP_ERROR else f"Warmup skipped: {WARMUP_ERROR}"
    final_status = [
        "Generated with Kokoro on CPU.",
        f"Voice: {voice}",
        f"Accent: {'British' if voice.startswith('b') else 'American'}",
        f"Speed: {speed:.2f}",
        f"Saved: {output_path}",
        warm_status,
        enhance_message,
    ]
    progress(1.0, desc="Complete")
    yield str(output_path), "\n".join(final_status), voice_profile(voice)


with gr.Blocks(title="Studio Voice") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.HTML(
        """
        <div class="app-shell">
          <div class="chrome-bar">
            <div class="brand-lockup">
              <span class="brand-mark"></span>
              <span>Studio Voice</span>
            </div>
            <div class="chrome-links">
              <span>CPU engine</span>
              <span>24kHz render</span>
              <span>Kokoro voices</span>
            </div>
          </div>
          <div class="hero-panel">
            <div class="hero-grid">
              <div>
                <div class="hero-kicker">Voice Rendering Console</div>
                <h1 class="hero-title">Minimal surface. Cinematic lighting. Premium voice in one place.</h1>
                <p class="hero-copy">The redesign pulls from your reference image: black glass panels, controlled highlight sweeps, soft shadows, subtle dot texture, and a tighter, more luxurious dashboard feel without changing the generator underneath.</p>
                <div class="metric-row">
                  <div class="metric-card">
                    <div class="metric-label">Mode</div>
                    <div class="metric-value">CPU-only studio stack</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Output</div>
                    <div class="metric-value">24kHz WAV playback</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-label">Launch</div>
                    <div class="metric-value">Desktop app shortcut</div>
                  </div>
                </div>
              </div>
              <div class="hero-side">
                <div class="hero-side-label">Current posture</div>
                <div class="hero-side-value">Clean enough for daily use, dramatic enough to feel premium.</div>
                <div class="hero-side-copy">Type, choose a pretrained voice, tune the speed, and render from the same panel. The interface stays quieter so the voice stays central.</div>
              </div>
            </div>
          </div>
        </div>
        """
    )

    with gr.Row(elem_id="workbench-grid"):
        with gr.Column(scale=8, elem_id="left-column"):
            with gr.Group(elem_id="script-panel"):
                gr.HTML(
                    """
                    <div class="section-head">
                      <span>Compose</span>
                      <strong>Script Console</strong>
                    </div>
                    """
                )
                text_input = gr.Textbox(
                    label="Script",
                    show_label=False,
                    lines=13,
                    placeholder="Write narration, promo copy, a lesson intro, prayer, ad read, or trailer lines here...",
                    container=False,
                    elem_id="script-input",
                )
                generate_button = gr.Button("Generate Studio Voice", variant="primary", size="lg", elem_id="generate-btn")

            with gr.Row(elem_id="results-grid"):
                with gr.Group(elem_id="audio-panel"):
                    gr.HTML(
                        """
                        <div class="section-head">
                          <span>Playback</span>
                          <strong>Output</strong>
                        </div>
                        """
                    )
                    audio_output = gr.Audio(
                        label="Studio Output",
                        show_label=False,
                        type="filepath",
                        autoplay=True,
                        elem_id="audio-output",
                    )
                with gr.Group(elem_id="status-panel"):
                    gr.HTML(
                        """
                        <div class="section-head">
                          <span>Session</span>
                          <strong>Render Log</strong>
                        </div>
                        """
                    )
                    status_output = gr.Textbox(
                        label="Render Status",
                        show_label=False,
                        lines=8,
                        container=False,
                        elem_id="status-output",
                    )

            with gr.Group(elem_id="library-panel"):
                gr.HTML(voice_library_html())

        with gr.Column(scale=5, elem_id="right-column"):
            with gr.Group(elem_id="control-panel"):
                gr.HTML(
                    """
                    <div class="section-head">
                      <span>Tune</span>
                      <strong>Voice Controls</strong>
                    </div>
                    """
                )
                voice_input = gr.Dropdown(choices=VOICE_OPTIONS, value=DEFAULT_VOICE, label="Pretrained Voice")
                speed_input = gr.Slider(minimum=0.8, maximum=1.2, value=1.0, step=0.01, label="Speed")
                enhance_input = gr.Checkbox(
                    label="Try extra clarity pass after generation",
                    value=False,
                    info="If the Windows enhancement dependency is unavailable, the app still renders normally and explains the fallback.",
                )
            profile_output = gr.HTML(value=voice_profile(DEFAULT_VOICE))

    with gr.Group(elem_id="examples-panel"):
        gr.HTML(
            """
            <div class="section-head">
              <span>Quick starts</span>
              <strong>Prompt Presets</strong>
            </div>
            """
        )
        gr.Examples(
            examples=[
                ["Welcome back. In this lesson, we will break down the exact steps you need to follow.", DEFAULT_VOICE, 1.0, False],
                ["This product was designed to save your team hours every single week.", "am_michael", 0.98, False],
                ["Tonight, the city holds its breath as the final signal begins to glow.", "bm_fable", 0.92, False],
            ],
            inputs=[text_input, voice_input, speed_input, enhance_input],
        )

    voice_input.change(fn=voice_profile, inputs=voice_input, outputs=profile_output)
    generate_button.click(
        fn=synthesize,
        inputs=[text_input, voice_input, speed_input, enhance_input],
        outputs=[audio_output, status_output, profile_output],
    )


start_background_warmup()


if __name__ == "__main__":
    import webview
    import webbrowser
    
    # Check if a server is already running to avoid port conflicts
    try:
        # Launch Gradio server in the background
        _, _, share_url = demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, prevent_thread_lock=True)
        
        try:
            # Create a native window pointing to the local Gradio server
            print("Creating desktop window...")
            window = webview.create_window(
                "Studio Voice",
                "http://127.0.0.1:7860",
                width=1280,
                height=920,
                background_color="#030303",
                min_size=(800, 600)
            )
            
            # Start the webview (this blocks until the window is closed)
            print("Starting desktop GUI...")
            webview.start()
        except Exception as e:
            # Fallback to browser if webview fails (e.g., missing dependencies)
            print(f"Native window failed to start: {e}")
            webbrowser.open("http://127.0.0.1:7860")
            # Wait for user to manually close the server or just keep it running
            threading.Event().wait() 
            
    except Exception as e:
        print(f"Failed to launch Studio Voice server: {e}")
    finally:
        demo.close()
        os._exit(0)

