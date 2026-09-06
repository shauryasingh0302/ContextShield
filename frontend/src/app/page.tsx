"use client";

import { useState, useRef, ChangeEvent, DragEvent, FormEvent, ReactNode } from "react";
import Link from "next/link";

interface UploadResponse {
  success?: boolean;
  status?: string;
  message?: string;
  risk_label: "SAFE" | "OFFENSIVE" | "HATE" | "HARASSMENT" | string;
  risk_score: number;
  confidence: number;
  probabilities: {
    SAFE: number;
    OFFENSIVE: number;
    HATE: number;
    HARASSMENT: number;
  };
  detected_text?: string;
  ocr_text?: string;
  explanation?: string | null;
  suggestion?: string | null;
  caption?: string;
  text_analysis?: {
    caption_embedding_size: number;
    ocr_embedding_size: number;
  };
  image_analysis?: {
    embedding_size: number;
  };
  multimodal_analysis?: {
    caption_features: number;
    ocr_features: number;
    image_features: number;
    fusion_features: number;
  };
  data?: {
    saved_filename?: string;
    original_filename?: string;
    content_type?: string;
    size_bytes?: number;
    caption?: string;
    ocr_text?: string;
    risk_label?: string;
    risk_score?: number;
    confidence?: number;
    probabilities?: {
      SAFE: number;
      OFFENSIVE: number;
      HATE: number;
      HARASSMENT: number;
    };
    explanation?: string | null;
    suggestion?: string | null;
    text_analysis?: {
      caption_embedding_size: number;
      ocr_embedding_size: number;
    };
    image_analysis?: {
      embedding_size: number;
    };
    multimodal_analysis?: {
      caption_features: number;
      ocr_features: number;
      image_features: number;
      fusion_features: number;
    };
  };
}

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".webp"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

interface VerdictConfig {
  title: string;
  badge: string;
  summary: string;
  icon: (cls: string) => ReactNode;
}

const VERDICT_CONFIGS: Record<string, VerdictConfig> = {
  SAFE: {
    title: "SAFE",
    badge: "LOW RISK",
    summary: "No significant safety risks detected across image and text context.",
    icon: (cls) => (
      <svg className={cls} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
  OFFENSIVE: {
    title: "OFFENSIVE",
    badge: "MODERATE RISK",
    summary: "May contain vulgar, derogatory, or insulting language or imagery.",
    icon: (cls) => (
      <svg className={cls} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
  },
  HATE: {
    title: "HATE SPEECH",
    badge: "HIGH RISK",
    summary: "Flagged for hateful or identity-targeted content against protected groups.",
    icon: (cls) => (
      <svg className={cls} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
      </svg>
    ),
  },
  HARASSMENT: {
    title: "HARASSMENT",
    badge: "HIGH RISK",
    summary: "Flagged for targeted personal attacks, bullying, or insults aimed at an individual.",
    icon: (cls) => (
      <svg className={cls} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
  },
};

const DEFAULT_VERDICT: VerdictConfig = {
  title: "EVALUATED",
  badge: "ANALYSIS COMPLETE",
  summary: "Post evaluated by multimodal classifier.",
  icon: (cls) => (
    <svg className={cls} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSelectFile = (selectedFile: File) => {
    setError(null);
    setResult(null);

    const fileExt = selectedFile.name.substring(selectedFile.name.lastIndexOf(".")).toLowerCase();
    const isExtensionAllowed = ALLOWED_EXTS.includes(fileExt);
    const isMimeAllowed = ALLOWED_TYPES.includes(selectedFile.type.toLowerCase());

    if (!isExtensionAllowed || !isMimeAllowed) {
      setError("Invalid file format. Please upload a JPG, JPEG, PNG, or WEBP image.");
      return;
    }

    if (selectedFile.size > MAX_SIZE_BYTES) {
      setError("File size exceeds 10 MB limit. Please select a smaller image.");
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelectFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelectFile(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image to analyze.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("caption", caption);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const res = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to analyze post. Server returned an error.");
      }

      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while connecting to the server.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    handleRemoveFile();
    setCaption("");
    setResult(null);
    setError(null);
  };

  const handleCopySuggestion = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const verdict = result?.risk_label ? (VERDICT_CONFIGS[result.risk_label] || DEFAULT_VERDICT) : DEFAULT_VERDICT;
  const detectedText = result?.detected_text ?? result?.ocr_text ?? "";

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 font-sans selection:bg-neutral-900 selection:text-white dark:selection:bg-white dark:selection:text-neutral-900">
      {/* 1. Header Navigation Bar */}
      <header className="shrink-0 h-13 border-b border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 z-20">
        <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-6 h-6 rounded bg-neutral-900 dark:bg-white flex items-center justify-center text-white dark:text-neutral-900">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <span className="font-semibold text-sm tracking-tight text-neutral-900 dark:text-neutral-100">
              ContextShield
            </span>
            <span className="hidden sm:inline-block text-[10px] font-mono px-1.5 py-0.5 rounded border border-neutral-200 dark:border-neutral-800 text-neutral-500">
              Multimodal Safety
            </span>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <Link
              href="/annotate"
              className="px-2.5 py-1 rounded border border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 dark:hover:border-neutral-600 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition flex items-center space-x-1"
            >
              <span>Dataset Annotator</span>
              <span className="text-neutral-400">→</span>
            </Link>
          </div>
        </div>
      </header>

      {/* 2. Main Workstation (Strict No-Scroll 2-Column Desktop Grid) */}
      <main className="flex-1 min-h-0 max-w-7xl w-full mx-auto p-3 sm:p-4 lg:p-5 grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 overflow-y-auto lg:overflow-hidden">
        {/* Left Column: Post Composer & Input Panel */}
        <section className="lg:col-span-5 flex flex-col min-h-0 bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-800 p-4 sm:p-5 shadow-sm overflow-y-auto space-y-4">
          <div className="flex items-center justify-between border-b border-neutral-100 dark:border-neutral-800 pb-2.5">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-neutral-900 dark:text-neutral-100">
                Post Composer
              </h2>
              <p className="text-[11px] text-neutral-500">Input image and caption to evaluate safety</p>
            </div>
            {result && (
              <button
                type="button"
                onClick={handleReset}
                className="text-[11px] font-mono px-2 py-0.5 rounded border border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100 transition"
              >
                Clear
              </button>
            )}
          </div>

          {error && (
            <div className="p-3 rounded border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 flex items-start justify-between text-xs">
              <span className="leading-snug">{error}</span>
              <button onClick={() => setError(null)} className="text-neutral-400 hover:text-neutral-700 font-mono ml-2">
                ✕
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex-1 flex flex-col space-y-4">
            {/* Image Upload Zone */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-neutral-600 dark:text-neutral-400">1. Image / Meme</span>
                <span className="text-[10px] font-mono text-neutral-400">Max 10MB</span>
              </div>

              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                className="hidden"
              />

              {!previewUrl ? (
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border border-dashed rounded-lg p-5 text-center cursor-pointer transition-colors ${
                    isDragging
                      ? "border-neutral-900 dark:border-neutral-100 bg-neutral-100/60 dark:bg-neutral-800/40"
                      : "border-neutral-300 dark:border-neutral-700 hover:border-neutral-500 bg-neutral-50/50 dark:bg-neutral-950/40"
                  }`}
                >
                  <div className="flex flex-col items-center justify-center space-y-2 text-neutral-500 dark:text-neutral-400">
                    <svg className="w-5 h-5 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="text-xs">
                      <span className="font-semibold text-neutral-900 dark:text-neutral-100 underline underline-offset-2">Click to select image</span> or drag here
                    </p>
                    <p className="text-[10px] text-neutral-400">JPG, PNG, or WEBP</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="relative rounded-lg overflow-hidden border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-950 flex items-center justify-center max-h-48 sm:max-h-56">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Uploaded Post Draft"
                      className="max-h-48 sm:max-h-56 w-auto object-contain rounded"
                    />
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-neutral-500 dark:text-neutral-400 bg-neutral-50 dark:bg-neutral-950 px-2.5 py-1.5 rounded border border-neutral-200 dark:border-neutral-800">
                    <span className="truncate max-w-[180px] font-mono text-neutral-700 dark:text-neutral-300">
                      {file?.name} ({file ? formatFileSize(file.size) : ""})
                    </span>
                    <div className="flex items-center space-x-2">
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="text-neutral-900 dark:text-neutral-100 hover:underline font-medium"
                      >
                        Change
                      </button>
                      <span>/</span>
                      <button
                        type="button"
                        onClick={handleRemoveFile}
                        className="text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Caption Textarea */}
            <div className="space-y-1.5 flex-1 flex flex-col">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-neutral-600 dark:text-neutral-400">2. Caption</span>
                <span className="text-[10px] font-mono text-neutral-400">{caption.length} chars</span>
              </div>
              <textarea
                id="caption"
                rows={3}
                value={caption}
                onChange={(e) => setCaption(e.target.value)}
                placeholder="Enter caption accompanying the image..."
                className="w-full flex-1 min-h-[72px] rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 p-3 text-xs text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-none focus:border-neutral-400 dark:focus:border-neutral-600 transition resize-none"
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={!file || isLoading}
              className="w-full py-2.5 px-4 rounded-lg font-medium text-xs text-white dark:text-neutral-900 bg-neutral-900 dark:bg-white hover:bg-black dark:hover:bg-neutral-200 active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Analyzing Post Context...</span>
                </>
              ) : (
                <span>Analyze Post Safety</span>
              )}
            </button>
          </form>
        </section>

        {/* Right Column: Analysis Results & Findings */}
        <section className="lg:col-span-7 flex flex-col min-h-0 bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-800 p-4 sm:p-5 shadow-sm overflow-y-auto space-y-5">
          {!result ? (
            /* Standby State */
            <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4 my-auto">
              <div className="w-12 h-12 rounded border border-neutral-300 dark:border-neutral-700 flex items-center justify-center text-neutral-400">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div className="space-y-1 max-w-sm">
                <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 tracking-tight">
                  Analysis Engine Ready
                </h3>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 leading-relaxed">
                  Upload an image and caption on the left to evaluate safety against hate speech, harassment, and offensive content.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-2 max-w-sm w-full pt-2 text-[11px] font-mono text-neutral-500">
                <div className="p-2 rounded border border-neutral-100 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-left">
                  <span className="font-bold text-neutral-800 dark:text-neutral-200">SAFE</span>
                  <div className="text-[10px] text-neutral-400">Non-harmful post</div>
                </div>
                <div className="p-2 rounded border border-neutral-100 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-left">
                  <span className="font-bold text-neutral-800 dark:text-neutral-200">OFFENSIVE</span>
                  <div className="text-[10px] text-neutral-400">Vulgar / insulting</div>
                </div>
                <div className="p-2 rounded border border-neutral-100 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-left">
                  <span className="font-bold text-neutral-800 dark:text-neutral-200">HATE SPEECH</span>
                  <div className="text-[10px] text-neutral-400">Group-targeted hate</div>
                </div>
                <div className="p-2 rounded border border-neutral-100 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-left">
                  <span className="font-bold text-neutral-800 dark:text-neutral-200">HARASSMENT</span>
                  <div className="text-[10px] text-neutral-400">Individual bullying</div>
                </div>
              </div>
            </div>
          ) : (
            /* Active Analysis Results */
            <div className="space-y-4">
              {/* Verdict Header */}
              <div className="flex items-start justify-between border-b border-neutral-100 dark:border-neutral-800 pb-3.5">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded border border-neutral-300 dark:border-neutral-700 flex items-center justify-center flex-shrink-0 text-neutral-900 dark:text-neutral-100">
                    {verdict.icon("w-5 h-5")}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] font-mono uppercase tracking-widest text-neutral-400">Verdict</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-800 text-neutral-800 dark:text-neutral-200">
                        {verdict.badge}
                      </span>
                    </div>
                    <h2 className="text-xl font-bold tracking-tight text-neutral-950 dark:text-neutral-50">
                      {verdict.title}
                    </h2>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={handleReset}
                  className="px-2.5 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:border-neutral-500 text-[11px] font-medium text-neutral-700 dark:text-neutral-300 transition"
                >
                  Analyze Another
                </button>
              </div>

              {/* Score & Confidence Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 space-y-1.5">
                  <div className="flex justify-between text-[11px] text-neutral-500">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">Risk Score</span>
                    <span className="font-mono">{result.risk_score?.toFixed(1)} / 100</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-neutral-950 dark:text-neutral-50">
                    {result.risk_score?.toFixed(1)}%
                  </div>
                  <div className="w-full h-1 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
                    <div
                      className="h-full bg-neutral-900 dark:bg-neutral-100 transition-all duration-500"
                      style={{ width: `${Math.min(100, Math.max(0, result.risk_score))}%` }}
                    />
                  </div>
                </div>

                <div className="p-3 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 space-y-1.5">
                  <div className="flex justify-between text-[11px] text-neutral-500">
                    <span className="font-semibold uppercase tracking-wider text-[10px]">Confidence</span>
                    <span className="font-mono">Certainty</span>
                  </div>
                  <div className="text-xl font-bold font-mono text-neutral-950 dark:text-neutral-50">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="w-full h-1 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
                    <div
                      className="h-full bg-neutral-900 dark:bg-neutral-100 transition-all duration-500"
                      style={{ width: `${Math.min(100, Math.max(0, result.confidence * 100))}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* 4-Class Probability Breakdown */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                  <span>Class Probability Distribution</span>
                  <span className="font-mono">4 Classes</span>
                </div>
                <div className="p-3 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/30 dark:bg-neutral-950/30 space-y-2">
                  {(["SAFE", "OFFENSIVE", "HATE", "HARASSMENT"] as const).map((cat) => {
                    const prob = result.probabilities?.[cat] ?? 0;
                    const pct = (prob * 100).toFixed(1);
                    const isMatch = result.risk_label === cat;

                    return (
                      <div key={cat} className="space-y-1">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className={`font-medium ${isMatch ? "text-neutral-950 dark:text-white font-bold" : "text-neutral-500"}`}>
                            {cat} {isMatch && <span className="ml-1 text-[9px] font-mono text-neutral-400">●</span>}
                          </span>
                          <span className="font-mono text-neutral-600 dark:text-neutral-400">{pct}%</span>
                        </div>
                        <div className="w-full h-1 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${isMatch ? "bg-neutral-900 dark:bg-white" : "bg-neutral-400 dark:bg-neutral-600"}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Explanation */}
              <div className="space-y-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                  Analysis Explanation
                </span>
                <div className="p-3 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-xs text-neutral-800 dark:text-neutral-200 leading-relaxed">
                  {result.explanation || "No explanation provided for this evaluation."}
                </div>
              </div>

              {/* Safer Caption Suggestion */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                    Safer Caption Suggestion
                  </span>
                  {result.suggestion && (
                    <button
                      type="button"
                      onClick={() => handleCopySuggestion(result.suggestion!)}
                      className="text-[11px] font-medium text-neutral-600 dark:text-neutral-400 hover:text-neutral-950 dark:hover:text-white transition"
                    >
                      {copied ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>

                {result.suggestion ? (
                  <div className="p-3 rounded border border-neutral-300 dark:border-neutral-700 bg-neutral-100/60 dark:bg-neutral-800/50 space-y-1">
                    <p className="text-xs font-medium text-neutral-900 dark:text-neutral-100 leading-relaxed">
                      "{result.suggestion}"
                    </p>
                  </div>
                ) : (
                  <div className="p-2.5 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 text-xs text-neutral-600 dark:text-neutral-400 flex items-center space-x-2">
                    <span className="font-mono text-neutral-500">[✓]</span>
                    <span>No changes needed. Post appears suitable for general publication.</span>
                  </div>
                )}
              </div>

              {/* OCR Extracted Text */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center text-[10px] font-semibold uppercase tracking-wider text-neutral-500">
                  <span>Extracted Image Text (OCR)</span>
                  <span className="font-mono lowercase text-neutral-400">
                    {detectedText.trim() ? `${detectedText.trim().length} chars` : "0 chars"}
                  </span>
                </div>
                {detectedText.trim() ? (
                  <div className="p-2.5 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/40 font-mono text-[11px] text-neutral-800 dark:text-neutral-200 whitespace-pre-wrap leading-relaxed">
                    {detectedText}
                  </div>
                ) : (
                  <div className="p-2.5 rounded border border-neutral-200 dark:border-neutral-800 bg-neutral-50/30 dark:bg-neutral-950/30 text-xs text-neutral-400 italic">
                    No text detected in this image.
                  </div>
                )}
              </div>

              {/* Collapsible Diagnostics */}
              <details className="text-[10px] font-mono border-t border-neutral-100 dark:border-neutral-800 pt-3 text-neutral-500 group">
                <summary className="cursor-pointer select-none font-sans font-medium text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 flex items-center justify-between">
                  <span>Multimodal Pipeline Details</span>
                  <span className="group-open:rotate-180 transition-transform">▾</span>
                </summary>
                <div className="pt-2 space-y-1 text-[10px] leading-relaxed">
                  <div>Caption Embedding: {result.text_analysis?.caption_embedding_size ?? 768}d (XLM-RoBERTa)</div>
                  <div>OCR Embedding: {result.text_analysis?.ocr_embedding_size ?? 768}d (XLM-RoBERTa)</div>
                  <div>Image Embedding: {result.image_analysis?.embedding_size ?? 512}d (CLIP ViT-B/32)</div>
                  <div>Multimodal Fusion: {result.multimodal_analysis?.fusion_features ?? 256}d Gated Representation</div>
                  <div>Classifier: checkpoints/best_model.pt (Test Macro-F1: 76.5%)</div>
                </div>
              </details>
            </div>
          )}
        </section>
      </main>

      {/* 3. Understated Footer */}
      <footer className="shrink-0 h-8 border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 flex items-center justify-between px-4 sm:px-6 text-[11px] text-neutral-500 font-mono">
        <span>ContextShield • Multimodal Safety</span>
        <span className="hidden sm:inline">Academic Research Project</span>
      </footer>
    </div>
  );
}
