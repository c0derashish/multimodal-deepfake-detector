import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const THEME_KEY = "deepfake-detector-theme";

const LABEL_META = {
  FAKE: {
    tone: "danger",
    badge: "Likely manipulated",
    icon: "!",
    summary: "The model found coordinated signals that suggest tampering in the video.",
  },
  REAL: {
    tone: "success",
    badge: "Likely authentic",
    icon: "OK",
    summary: "The analysed modalities align more closely with an authentic recording.",
  },
  UNCERTAIN: {
    tone: "warning",
    badge: "Needs review",
    icon: "?",
    summary: "The system saw mixed evidence and could not reach a confident conclusion.",
  },
  ERROR: {
    tone: "neutral",
    badge: "Unavailable",
    icon: "X",
    summary: "The analysis did not complete successfully.",
  },
};

const PROGRESS_STEPS = [
  "Uploading secure sample",
  "Extracting video frames and audio",
  "Scanning facial inconsistencies",
  "Checking vocal artefacts",
  "Reviewing transcript and sync cues",
  "Fusing multimodal evidence",
  "Preparing explanation panels",
];

const pct = (value = 0) => `${(value * 100).toFixed(1)}%`;

const formatSeconds = (value) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}s` : "N/A";

const formatResolution = (meta) =>
  meta?.width && meta?.height ? `${meta.width} x ${meta.height}` : "N/A";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function apiUrl(path) {
  return API ? `${API}${path}` : path;
}

function normalizeError(error) {
  if (error instanceof Error && error.name === "TypeError") {
    return "Cannot reach the backend API. Make sure the FastAPI server is running and that the frontend is using the correct API URL.";
  }

  return error instanceof Error ? error.message : "Something went wrong";
}

async function readErrorMessage(response, fallbackMessage) {
  try {
    const data = await response.json();
    return data?.detail || data?.error || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

function toneForScore(score = 0) {
  if (score >= 0.6) return "danger";
  if (score >= 0.4) return "warning";
  return "success";
}

function UploadZone({ onFile, fileName }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const acceptFile = (file) => {
    if (file) onFile(file);
  };

  return (
    <button
      type="button"
      className={`upload-zone ${dragging ? "upload-zone--dragging" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        acceptFile(event.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        className="sr-only"
        type="file"
        accept="video/*"
        onChange={(event) => acceptFile(event.target.files?.[0])}
      />
      <div className="upload-zone__icon" aria-hidden="true">
        <span class="material-symbols-outlined">
          attachment
        </span>
      </div>
      <p className="eyebrow">Upload sample</p>
      <h2>Drop a video or browse from your device</h2>
      <p className="upload-zone__copy">
        Built for quick deepfake triage across video, audio, and transcript signals.
      </p>
      <div className="upload-zone__meta">
        <span>Formats: MP4, MOV, AVI, MKV, WebM</span>
        <span>Recommended: under 500 MB</span>
      </div>
      {fileName ? <p className="upload-zone__filename">Selected: {fileName}</p> : null}
    </button>
  );
}

function MetricCard({ label, value, helper }) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {helper ? <span className="metric-card__helper">{helper}</span> : null}
    </div>
  );
}

function ProgressPanel({ step, fileName, jobId }) {
  const index = Math.min(step, PROGRESS_STEPS.length - 1);
  const progress = Math.max(12, Math.round(((index + 1) / PROGRESS_STEPS.length) * 100));

  return (
    <section className="panel panel--processing">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Pipeline running</p>
          <h2>Analysing {fileName || "your video"}</h2>
        </div>
        <div className="status-pill status-pill--active">In progress</div>
      </div>

      <div className="progress-ring" style={{ "--p": progress }}>
        <div className="progress-ring__value">{progress}%</div>
      </div>

      <div className="progress-track" aria-hidden="true">
        <div className="progress-track__fill" style={{ width: `${progress}%` }} />
      </div>

      <p className="progress-step">{PROGRESS_STEPS[index]}</p>

      <div className="processing-meta">
        <span>Typical runtime: 30 to 120 seconds</span>
        {jobId ? <span>Job ID: {jobId}</span> : null}
      </div>
    </section>
  );
}

function Verdict({ result }) {
  const meta = LABEL_META[result.label] || LABEL_META.ERROR;

  return (
    <section className={`panel verdict verdict--${meta.tone}`}>
      <div className="verdict__topline">
        <span className={`status-pill status-pill--${meta.tone}`}>{meta.badge}</span>
        <span className="verdict__runtime">Processed in {formatSeconds(result.processing_time_s)}</span>
      </div>

      <div className="verdict__main">
        <div className="verdict__icon" aria-hidden="true">
          {meta.icon}
        </div>
        <div>
          <p className="eyebrow">Primary verdict</p>
          <h2>{result.label}</h2>
          <p className="verdict__summary">{meta.summary}</p>
        </div>
      </div>

      <div className="verdict__stats">
        <MetricCard label="Fake probability" value={pct(result.fake_probability)} />
        <MetricCard label="Confidence" value={result.confidence || "N/A"} />
        <MetricCard label="Processing time" value={formatSeconds(result.processing_time_s)} />
      </div>
    </section>
  );
}

function SignalBar({ label, score, icon }) {
  const tone = toneForScore(score);

  return (
    <div className="signal-row">
      <div className="signal-row__header">
        <div className="signal-row__label">
          <span className={`signal-dot signal-dot--${tone}`} aria-hidden="true" />
          <span>{icon}</span>
          <span>{label}</span>
        </div>
        <strong>{pct(score)}</strong>
      </div>
      <div className="signal-bar">
        <div className={`signal-bar__fill signal-bar__fill--${tone}`} style={{ width: pct(score) }} />
      </div>
    </div>
  );
}

function ModalityBreakdown({ result }) {
  const modalities = [
    { label: "Vision analysis", key: "video_score", icon: "Video" },
    { label: "Audio analysis", key: "audio_score", icon: "Audio" },
    { label: "Text and lip-sync", key: "text_score", icon: "Sync" },
  ];

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Evidence map</p>
          <h3>Modality breakdown</h3>
        </div>
      </div>

      <div className="signal-grid">
        {modalities.map(({ label, key, icon }) => (
          <SignalBar key={key} label={label} icon={icon} score={result[key] ?? 0} />
        ))}
      </div>

      {result.breakdown_plot ? (
        <div className="chart-frame">
          <img
            src={`data:image/png;base64,${result.breakdown_plot}`}
            alt="Modality breakdown chart"
          />
        </div>
      ) : null}
    </section>
  );
}

function AudioVisual({ result }) {
  if (!result.audio_plot) return null;

  const suspicious = result.modality_details?.audio?.fake_probability > 0.5;

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Waveform review</p>
          <h3>Audio forensics</h3>
        </div>
        {suspicious ? <div className="status-pill status-pill--danger">Suspicious segments</div> : null}
      </div>

      <div className="chart-frame">
        <img
          src={`data:image/png;base64,${result.audio_plot}`}
          alt="Audio waveform and spectrogram"
        />
      </div>

      <p className="panel__note">
        Use this panel to inspect abrupt transitions, spectral smoothing, and other synthetic voice cues.
      </p>

    </section>
  );
}

function SuspiciousFrames({ frames }) {
  const [selectedFrame, setSelectedFrame] = useState(null);
 
  if (!frames?.length) return null;
 
  // Sort frames by score (highest suspicion first)
  const sortedFrames = [...frames].sort((a, b) => b.score - a.score);
  
  // Calculate statistics
  const avgScore = frames.reduce((sum, f) => sum + f.score, 0) / frames.length;
  const maxScore = Math.max(...frames.map(f => f.score));
 
  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Frame review</p>
          <h3>Suspicious frames</h3>
        </div>
        <div className="status-pill status-pill--danger">{frames.length} flagged</div>
      </div>
 
      <div className="verdict__stats" style={{ marginBottom: '1.5rem' }}>
        <MetricCard label="Frames flagged" value={frames.length} />
        <MetricCard label="Average risk" value={pct(avgScore)} />
        <MetricCard label="Highest risk" value={pct(maxScore)} />
      </div>
 
      <p className="panel__note">
        Frames ranked by suspicion score. Look for facial warping, boundary artifacts, inconsistent lighting, or unnatural textures.
      </p>
 
      <div className="frame-grid">
        {sortedFrames.map((frame, idx) => {
          const tone = toneForScore(frame.score);
          return (
            <article 
              key={`${frame.frame_idx}-${idx}`} 
              className="frame-card"
              onClick={() => setSelectedFrame(frame)}
              style={{ cursor: 'pointer' }}
            >
              <img
                src={`data:image/jpeg;base64,${frame.image}`}
                alt={`Suspicious frame ${frame.frame_idx}`}
                className="frame-card__image"
              />
              <div className="frame-card__meta">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <strong>Frame {frame.frame_idx}</strong>
                  <span className={`signal-dot signal-dot--${tone}`} aria-hidden="true" />
                </div>
                <span style={{ fontWeight: 600, color: `var(--color-${tone})` }}>
                  {pct(frame.score)} risk
                </span>
              </div>
              {frame.timestamp !== undefined && (
                <div style={{ 
                  fontSize: '0.75rem', 
                  color: 'var(--color-text-secondary)', 
                  padding: '0 0.75rem 0.5rem'
                }}>
                  @ {frame.timestamp.toFixed(2)}s
                </div>
              )}
            </article>
          );
        })}
      </div>
 
      {selectedFrame && (
        <div 
          className="frame-modal"
          onClick={() => setSelectedFrame(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.9)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2rem',
          }}
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              background: 'var(--color-surface)',
              borderRadius: '1rem',
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '1rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ margin: 0 }}>Frame {selectedFrame.frame_idx}</h3>
                <p style={{ margin: '0.25rem 0 0', color: 'var(--color-text-secondary)' }}>
                  Risk score: {pct(selectedFrame.score)}
                </p>
              </div>
              <button 
                onClick={() => setSelectedFrame(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: 'var(--color-text-primary)',
                }}
              >
                ×
              </button>
            </div>
            <img
              src={`data:image/jpeg;base64,${selectedFrame.image}`}
              alt={`Frame ${selectedFrame.frame_idx} detail`}
              style={{
                maxWidth: '100%',
                maxHeight: 'calc(90vh - 10rem)',
                objectFit: 'contain',
                borderRadius: '0.5rem',
              }}
            />
          </div>
        </div>
      )}
    </section>
  );
}

function ExplanationPanel({ explanation }) {
  if (!explanation?.length) return null;

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">Reasoning trace</p>
          <h3>Why the model said this</h3>
        </div>
      </div>

      <div className="explanation-list">
        {explanation.slice(0, 10).map((line, index) => (
          <article key={`${line}-${index}`} className="explanation-item">
            <span>{String(index + 1).padStart(2, "0")}</span>
            <p>{line}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function VideoMeta({ meta }) {
  if (!meta || !Object.keys(meta).length) return null;

  return (
    <section className="meta-grid">
      <MetricCard label="Duration" value={formatSeconds(meta.duration_s)} />
      <MetricCard label="Resolution" value={formatResolution(meta)} />
      <MetricCard label="Frame rate" value={meta.fps ? `${meta.fps.toFixed(0)} fps` : "N/A"} />
    </section>
  );
}

function ErrorPanel({ error, onReset }) {
  return (
    <section className="panel panel--error">
      <div className="status-pill status-pill--danger">Analysis failed</div>
      <h2>We hit a processing issue</h2>
      <p className="panel__note">{error}</p>
      <button type="button" className="primary-button" onClick={onReset}>
        Try another video
      </button>
    </section>
  );
}

function getInitialTheme() {
  if (typeof window === "undefined") {
    return "dark";
  }

  const savedTheme = window.localStorage.getItem(THEME_KEY);
  if (savedTheme === "light" || savedTheme === "dark") {
    return savedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [phase, setPhase] = useState("idle");
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(0);
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.body.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const reset = () => {
    setPhase("idle");
    setFile(null);
    setJobId(null);
    setResult(null);
    setError(null);
    setStep(0);
  };

  const handleFile = async (selectedFile) => {
    setFile(selectedFile);
    setPhase("uploading");
    setError(null);
    setResult(null);
    setStep(0);

    try {
      const form = new FormData();
      form.append("file", selectedFile);

      const uploadResponse = await fetch(apiUrl("/api/upload"), {
        method: "POST",
        body: form,
      });

      if (!uploadResponse.ok) {
        throw new Error(await readErrorMessage(uploadResponse, "Upload failed"));
      }

      const { job_id } = await uploadResponse.json();
      setJobId(job_id);
      setStep(1);
      setPhase("polling");

      const analyseResponse = await fetch(apiUrl(`/api/analyze/${job_id}`), {
        method: "POST",
      });

      if (!analyseResponse.ok) {
        throw new Error(await readErrorMessage(analyseResponse, "Unable to start analysis"));
      }

      const stepTimings = [2500, 3000, 3500, 2500, 2000];
      for (let index = 0; index < stepTimings.length; index += 1) {
        await sleep(stepTimings[index]);
        setStep(index + 2);
      }

      let attempts = 0;
      while (attempts < 120) {
        const response = await fetch(apiUrl(`/api/result/${job_id}`));
        if (!response.ok) {
          throw new Error(await readErrorMessage(response, "Unable to fetch analysis result"));
        }
        const data = await response.json();

        if (data.status === "done") {
          setResult(data);
          setPhase("done");
          return;
        }

        if (data.status === "error") {
          throw new Error(data.error || "Analysis failed");
        }

        await sleep(2000);
        attempts += 1;
      }

      throw new Error("Timed out waiting for result");
    } catch (err) {
      setError(normalizeError(err));
      setPhase("error");
    }
  };

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"));
  };

  return (
    <div className="app-shell">
      <div className="app-background" aria-hidden="true">
        <span className="orb orb--one" />
        <span className="orb orb--two" />
        <span className="grid-glow" />
      </div>

      <main className="app">
        <div className="app-toolbar">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-pressed={theme === "dark"}
          >
            <span className="theme-toggle__track" aria-hidden="true">
              <span className={`theme-toggle__thumb theme-toggle__thumb--${theme}`} />
              <span className={`theme-toggle__option ${theme === "light" ? "is-active" : ""}`}>
                Light
              </span>
              <span className={`theme-toggle__option ${theme === "dark" ? "is-active" : ""}`}>
                Dark
              </span>
            </span>
          </button>
        </div>

        <section className="hero">
          <div className="hero__copy">
            <p className="eyebrow mdd">Review suspicious media with a cleaner, faster forensic workspace.</p>
            <h1>Multimodal Deepfake Detector</h1>
            <p className="hero__lede">
              Upload a video and inspect the decision across facial behaviour, audio artefacts,
              and text alignment in one place.
            </p>
            <div className="hero__chips">
              <span>Video signals</span>
              <span>Audio forensics</span>
              <span>Transcript cues</span>
              <span>Explainable output</span>
            </div>
          </div>

          <aside className="hero__card panel">
            <p className="eyebrow">System snapshot</p>
            <h2>Designed for quick triage</h2>
            <div className="hero__stats">
              <MetricCard label="Pipeline" value="3 modalities" helper="Vision, audio, language" />
              <MetricCard label="Output" value="Verdict + evidence" helper="Plots and explanation" />
            </div>
          </aside>
        </section>

        <section className="workspace">
          <div className="workspace__main">
            {phase === "idle" ? (
              <UploadZone onFile={handleFile} fileName={file?.name} />
            ) : null}

            {(phase === "uploading" || phase === "polling") && (
              <ProgressPanel step={step} fileName={file?.name} jobId={jobId} />
            )}

            {phase === "error" ? <ErrorPanel error={error} onReset={reset} /> : null}

            {phase === "done" && result ? (
              <div className="results-stack">
                <Verdict result={result} />
                <VideoMeta meta={result.video_metadata} />
                <ModalityBreakdown result={result} />
                <SuspiciousFrames frames={result.suspicious_frames} />
                <AudioVisual result={result} />
                <ExplanationPanel explanation={result.explanation} />
                <button type="button" className="primary-button primary-button--wide" onClick={reset}>
                  Analyse another video
                </button>
              </div>
            ) : null}
          </div>

          <aside className="workspace__side">
            <section className="panel side-panel">
              <p className="eyebrow">Workflow</p>
              <h3>What this app checks</h3>
              <div className="side-panel__list">
                <article>
                  <strong>01 Facial evidence</strong>
                  <p>Frame-level scanning for visual inconsistencies, expression drift, and artefacts.</p>
                </article>
                <article>
                  <strong>02 Voice evidence</strong>
                  <p>Waveform and spectrogram review to catch synthetic speech signatures.</p>
                </article>
                <article>
                  <strong>03 Language and sync</strong>
                  <p>Transcript-level and timing cues that help flag mismatched delivery.</p>
                </article>
              </div>
            </section>
          </aside>
        </section>

        <footer className="app-footer">Made with ❤️ by Code Monks</footer>
      </main>
    </div>
  );
}
