/* eslint-disable prettier/prettier */
import { useState } from "react";
import { Stethoscope } from "lucide-react";
import { useAuth } from "../lib/auth";

export function LoginScreen() {
  const { signIn } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(await signIn(username.trim(), password));
    setBusy(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-6">
      <form onSubmit={submit} className="w-full max-w-sm space-y-5 rounded-xl border border-border bg-card p-8 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground"><Stethoscope className="h-5 w-5" /></div>
          <div>
            <p className="text-lg font-bold leading-tight">TriageFlow</p>
            <p className="text-xs text-muted-foreground">Sign in to continue</p>
          </div>
        </div>
        <div>
          <label htmlFor="login-user" className="mb-1 block text-sm font-medium">Username</label>
          <input id="login-user" autoFocus autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        <div>
          <label htmlFor="login-pass" className="mb-1 block text-sm font-medium">Password</label>
          <input id="login-pass" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500" />
        </div>
        {error && <p role="alert" className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</p>}
        <button type="submit" disabled={busy || !username || !password}
          className="w-full rounded-md bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground disabled:opacity-50">
          {busy ? "Signing in..." : "Sign in"}
        </button>
        <p className="text-xs text-muted-foreground">Demo build with synthetic patients only. Accounts: rivera (charge nurse), lee (nurse), patel (physician), admin, viewer.</p>
      </form>
    </div>
  );
}
