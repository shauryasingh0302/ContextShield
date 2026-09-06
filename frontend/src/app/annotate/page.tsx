"use client";

import { useState, useRef, useEffect, ChangeEvent, FormEvent } from "react";
import Link from "next/link";

interface AnnotationSample {
  id: string;
  image: string;
  caption: string;
  label: string;
  severity: number;
  language: string;
}

const LABELS = [
  { value: "SAFE", label: "SAFE", desc: "Benign, non-harmful content" },
  { value: "OFFENSIVE", label: "OFFENSIVE", desc: "Vulgar, crude, profane" },
  { value: "HATE", label: "HATE SPEECH", desc: "Identity hate, slurs, incitement" },
  { value: "HARASSMENT", label: "HARASSMENT", desc: "Targeted bullying, threats" },
];

const SEVERITIES = [
  { value: 0, label: "0 - None", desc: "No risk (benign)" },
  { value: 1, label: "1 - Low", desc: "Mild profanity / borderline" },
  { value: 2, label: "2 - Moderate", desc: "Hostile / explicit abuse" },
  { value: 3, label: "3 - High", desc: "Severe hate / physical threats" },
];

const LANGUAGES = ["ENGLISH", "HINDI", "HINGLISH", "OTHER"];

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".webp"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

export default function AnnotatePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState<string>("");
  const [label, setLabel] = useState<string>("SAFE");
  const [severity, setSeverity] = useState<number>(0);
  const [language, setLanguage] = useState<string>("ENGLISH");

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const [totalCount, setTotalCount] = useState<number>(0);
  const [recentSamples, setRecentSamples] = useState<AnnotationSample[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchAnnotations = async () => {
    try {
      const res = await fetch("http://localhost:8000/annotations");
      if (res.ok) {
        const data = await res.json();
        setTotalCount(data.total || 0);
        setRecentSamples(data.samples || []);
      }
    } catch {
      // Backend may be offline during build
    }
  };

  useEffect(() => {
    fetchAnnotations();
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    setError(null);
    const fileExt = selectedFile.name.substring(selectedFile.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTS.includes(fileExt) || !ALLOWED_TYPES.includes(selectedFile.type.toLowerCase())) {
      setError("Invalid format. Please upload JPG, JPEG, PNG, or WEBP.");
      return;
    }
    if (selectedFile.size > MAX_SIZE_BYTES) {
      setError("File exceeds 10 MB limit.");
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
      handleFileSelect(e.target.files[0]);
    }
  };

  const resetForm = () => {
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setCaption("");
    setLabel("SAFE");
    setSeverity(0);
    setLanguage("ENGLISH");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMsg(null);

    if (!file) {
      setError("Please select an image to annotate.");
      return;
    }
    if (!caption.trim()) {
      setError("Please enter a caption for the image.");
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("caption", caption.trim());
      formData.append("label", label);
      formData.append("severity", severity.toString());
      formData.append("language", language);

      const res = await fetch("http://localhost:8000/annotate", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to save annotation.");
      }

      setSuccessMsg(`Sample ${data.id} saved to dataset/raw/annotations.jsonl`);
      resetForm();
      fetchAnnotations();
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while saving.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 font-sans selection:bg-neutral-900 selection:text-white dark:selection:bg-white dark:selection:text-neutral-900">
      {/* 1. Top Navigation Bar */}
      <header className="shrink-0 h-13 border-b border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 z-20">
        <div className="h-full max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="w-6 h-6 rounded bg-neutral-900 dark:bg-white flex items-center justify-center text-white dark:text-neutral-900">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <span className="font-semibold text-sm tracking-tight text-neutral-900 dark:text-neutral-100">
              ContextShield Annotator
            </span>
            <span className="hidden sm:inline-block text-[10px] font-mono px-2 py-0.5 rounded border border-neutral-200 dark:border-neutral-800 text-neutral-500">
              Total: {totalCount}
            </span>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <Link
              href="/"
              className="px-2.5 py-1 rounded border border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 dark:hover:border-neutral-600 text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 transition flex items-center space-x-1"
            >
              <span className="text-neutral-400">←</span>
              <span>Back to Analyzer</span>
            </Link>
          </div>
        </div>
      </header>

      {/* 2. Main Workstation (No-Scroll 2-Column Desktop Grid) */}
      <main className="flex-1 min-h-0 max-w-7xl w-full mx-auto p-3 sm:p-4 lg:p-5 grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5 overflow-y-auto lg:overflow-hidden">
        {/* Left Column: Media & Input Fields */}
        <section className="lg:col-span-5 flex flex-col min-h-0 bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-800 p-4 sm:p-5 shadow-sm overflow-y-auto space-y-4">
          <div className="border-b border-neutral-100 dark:border-neutral-800 pb-2.5">
            <h2 className="text-xs font-bold uppercase tracking-wider text-neutral-900 dark:text-neutral-100">
              1. Media & Context
            </h2>
            <p className="text-[11px] text-neutral-500">Upload source meme and accompanying text</p>
          </div>

          {error && (
            <div className="p-3 rounded border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 flex items-start justify-between text-xs">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-neutral-400 hover:text-neutral-700 font-mono ml-2">✕</button>
            </div>
          )}

          {successMsg && (
            <div className="p-3 rounded border border-neutral-300 dark:border-neutral-700 bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 flex items-start justify-between text-xs">
              <span>[✓] {successMsg}</span>
              <button onClick={() => setSuccessMsg(null)} className="text-neutral-400 hover:text-neutral-700 font-mono ml-2">✕</button>
            </div>
          )}

          {/* Image Upload Area */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-neutral-600 dark:text-neutral-400">
              Image / Meme
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border border-dashed border-neutral-300 dark:border-neutral-700 hover:border-neutral-500 rounded-lg p-3 flex flex-col items-center justify-center min-h-[140px] cursor-pointer bg-neutral-50/50 dark:bg-neutral-950/40 transition"
            >
              {previewUrl ? (
                <div className="relative w-full h-36 flex items-center justify-center overflow-hidden rounded">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={previewUrl} alt="Preview" className="max-h-full max-w-full object-contain" />
                </div>
              ) : (
                <div className="text-center space-y-1 text-neutral-500 dark:text-neutral-400">
                  <svg className="w-5 h-5 mx-auto text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-xs font-medium">Click to select image</p>
                  <p className="text-[10px] text-neutral-400">JPG, PNG, or WEBP (Max: 10 MB)</p>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
            {file && (
              <div className="flex items-center justify-between text-[11px] text-neutral-500 font-mono">
                <span className="truncate max-w-[200px]">{file.name}</span>
                <button
                  type="button"
                  onClick={() => { setFile(null); setPreviewUrl(null); }}
                  className="text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
                >
                  Remove
                </button>
              </div>
            )}
          </div>

          {/* Caption Textarea */}
          <div className="space-y-1.5 flex-1 flex flex-col">
            <label className="block text-xs font-semibold text-neutral-600 dark:text-neutral-400">
              Post Caption
            </label>
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Enter post caption or associated text..."
              rows={4}
              className="w-full flex-1 min-h-[80px] p-3 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 text-xs text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 focus:outline-none focus:border-neutral-400 transition resize-none"
            />
          </div>

          {/* Language Selection */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold text-neutral-600 dark:text-neutral-400">
              Language
            </label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full p-2 rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50/50 dark:bg-neutral-950/50 text-xs text-neutral-900 dark:text-neutral-100 focus:outline-none focus:border-neutral-400"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </select>
          </div>
        </section>

        {/* Right Column: Annotation Ground Truth & Recent Records */}
        <section className="lg:col-span-7 flex flex-col min-h-0 bg-white dark:bg-neutral-900 rounded-lg border border-neutral-200 dark:border-neutral-800 p-4 sm:p-5 shadow-sm overflow-y-auto space-y-4">
          <div className="border-b border-neutral-100 dark:border-neutral-800 pb-2.5 flex items-center justify-between">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-neutral-900 dark:text-neutral-100">
                2. Ground Truth Annotation
              </h2>
              <p className="text-[11px] text-neutral-500">Assign risk category and severity tier</p>
            </div>
            <button
              type="button"
              onClick={resetForm}
              className="text-[11px] font-mono px-2 py-0.5 rounded border border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100 transition"
            >
              Reset
            </button>
          </div>

          {/* Risk Category (4 Cards) */}
          <div className="space-y-1.5">
            <label className="block text-[11px] font-semibold uppercase tracking-wider text-neutral-500">
              Risk Category (Label)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {LABELS.map((item) => {
                const isSelected = label === item.value;
                return (
                  <button
                    key={item.value}
                    type="button"
                    onClick={() => setLabel(item.value)}
                    className={`p-2.5 rounded-lg border text-left transition ${
                      isSelected
                        ? "border-neutral-900 dark:border-neutral-100 bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                        : "border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-700 dark:text-neutral-300 bg-neutral-50/30 dark:bg-neutral-950/30"
                    }`}
                  >
                    <div className="text-xs font-bold">{item.label}</div>
                    <div className={`text-[10px] mt-0.5 ${isSelected ? "text-neutral-300 dark:text-neutral-600" : "text-neutral-400"}`}>
                      {item.desc}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Severity (4 Tiers) */}
          <div className="space-y-1.5">
            <label className="block text-[11px] font-semibold uppercase tracking-wider text-neutral-500">
              Severity Level
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {SEVERITIES.map((s) => {
                const isSelected = severity === s.value;
                return (
                  <button
                    key={s.value}
                    type="button"
                    onClick={() => setSeverity(s.value)}
                    className={`p-2 rounded border text-left transition ${
                      isSelected
                        ? "border-neutral-900 dark:border-neutral-100 bg-neutral-900 text-white dark:bg-white dark:text-neutral-900 font-bold"
                        : "border-neutral-200 dark:border-neutral-800 hover:border-neutral-400 text-neutral-600 dark:text-neutral-400"
                    }`}
                  >
                    <div className="text-[11px] font-mono font-semibold">{s.label}</div>
                    <div className={`text-[9px] mt-0.5 truncate ${isSelected ? "text-neutral-300 dark:text-neutral-600" : "text-neutral-400"}`}>
                      {s.desc}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isLoading || !file || !caption.trim()}
            className="w-full py-2.5 px-4 rounded-lg font-medium text-xs text-white dark:text-neutral-900 bg-neutral-900 dark:bg-white hover:bg-black dark:hover:bg-neutral-200 active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center space-x-2"
          >
            {isLoading ? <span>Saving Annotation...</span> : <span>Save Annotation to Dataset</span>}
          </button>

          {/* Recent Annotations Table */}
          {recentSamples.length > 0 && (
            <div className="space-y-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
              <div className="flex justify-between items-center text-[10px] font-mono text-neutral-400 uppercase tracking-wider">
                <span>Recent Annotations</span>
                <span>{recentSamples.length} records</span>
              </div>
              <div className="overflow-x-auto border border-neutral-200 dark:border-neutral-800 rounded">
                <table className="w-full text-left text-[11px]">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-800 text-neutral-400 bg-neutral-50/50 dark:bg-neutral-950/50">
                      <th className="p-2 font-mono">ID</th>
                      <th className="p-2">Caption</th>
                      <th className="p-2">Label</th>
                      <th className="p-2">Sev</th>
                      <th className="p-2">Lang</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800 font-mono text-[10px]">
                    {recentSamples.slice().reverse().map((s) => (
                      <tr key={s.id} className="hover:bg-neutral-50/50 dark:hover:bg-neutral-800/30">
                        <td className="p-2 font-bold text-neutral-900 dark:text-neutral-100">{s.id}</td>
                        <td className="p-2 max-w-[160px] truncate font-sans">{s.caption}</td>
                        <td className="p-2 font-bold">{s.label}</td>
                        <td className="p-2">{s.severity}</td>
                        <td className="p-2">{s.language}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* 3. Understated Footer */}
      <footer className="shrink-0 h-8 border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 flex items-center justify-between px-4 sm:px-6 text-[11px] text-neutral-500 font-mono">
        <span>ContextShield Annotator</span>
        <span className="hidden sm:inline">dataset/raw/annotations.jsonl</span>
      </footer>
    </div>
  );
}
