export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <div className="max-w-xl w-full text-center space-y-4 p-8 bg-white dark:bg-slate-900 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-800">
        <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-white">
          ContextShield
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-400">
          Check your social-media post before publishing.
        </p>
      </div>
    </main>
  );
}
