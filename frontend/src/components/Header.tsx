import { Bell, Bot, Search, Zap } from "lucide-react";
import { useTriage } from "../lib/triage-context";

export function Header() {
  const { aiActive, setAiActive, surge, setSurge, searchQuery, setSearchQuery } = useTriage();

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center gap-4 border-b border-border bg-card/95 px-6 backdrop-blur">
      <div className="relative w-full max-w-sm">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          type="search"
          placeholder="Search patient, MRN, or complaint…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="h-10 w-full rounded-lg border border-input bg-background pl-9 pr-3 text-sm outline-none transition-shadow placeholder:text-muted-foreground focus:ring-2 focus:ring-ring/30"
        />
      </div>

      <div className="ml-auto flex items-center gap-4">
        <div className="flex items-center gap-2.5 rounded-lg border border-border bg-background px-3 py-2">
          <Bot className={`h-4 w-4 ${aiActive ? "text-esi-4" : "text-muted-foreground"}`} />
          <span className={`text-xs font-semibold ${aiActive ? "text-foreground" : "text-muted-foreground"}`}>
            {aiActive ? "AI Active" : "Manual Mode"}
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={aiActive}
            aria-label="Toggle AI triage assistance"
            onClick={() => setAiActive(!aiActive)}
            className={`relative h-5.5 w-10 rounded-full transition-colors ${aiActive ? "bg-esi-4" : "bg-muted-foreground/40"}`}
          >
            <span
              className={`absolute top-0.5 h-4.5 w-4.5 rounded-full bg-white shadow transition-all ${
                aiActive ? "left-5" : "left-0.5"
              }`}
            />
          </button>
          {!aiActive && (
            <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-destructive">
              Kill Switch
            </span>
          )}
        </div>

        <button
          onClick={() => setSurge(!surge)}
          className={`inline-flex h-10 items-center gap-2 rounded-lg border px-4 text-sm font-semibold transition-colors ${
            surge
              ? "border-esi-2 bg-esi-2-soft text-foreground"
              : "border-border bg-card text-foreground hover:bg-muted"
          }`}
        >
          <Zap className="h-4 w-4" />
          <span className="whitespace-nowrap">{surge ? "Surge Active" : "Simulate Surge"}</span>
        </button>

        <button type="button" className="relative rounded-lg p-2 text-muted-foreground hover:bg-muted" aria-label="Notifications">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-esi-1" />
        </button>

        <div className="flex items-center gap-3 border-l border-border pl-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
            SR
          </div>
          <div className="hidden leading-tight lg:block">
            <p className="whitespace-nowrap text-sm font-semibold">Sarah Rivera, RN</p>
            <p className="text-xs text-muted-foreground">Charge Nurse</p>
          </div>
        </div>
      </div>
    </header>
  );
}
