/* eslint-disable prettier/prettier */
/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { describeError, fetchMe, getToken, login as apiLogin, setToken, setUnauthorizedHandler, type UserOut } from "./api";

interface AuthState {
  user: UserOut | null;
  /** false until we know whether a stored token is still valid */
  ready: boolean;
  signIn: (username: string, password: string) => Promise<string | null>;
  signOut: () => void;
  can: (permission: string) => boolean;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null);
  const [ready, setReady] = useState(false);

  const signOut = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(signOut);
    if (!getToken()) {
      setReady(true);
      return () => setUnauthorizedHandler(null);
    }
    fetchMe()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setReady(true));
    return () => setUnauthorizedHandler(null);
  }, [signOut]);

  const signIn = useCallback(async (username: string, password: string): Promise<string | null> => {
    try {
      const res = await apiLogin(username, password);
      setToken(res.access_token);
      setUser(res.user);
      return null;
    } catch (err) {
      return describeError(err);
    }
  }, []);

  const value = useMemo<AuthState>(
    () => ({ user, ready, signIn, signOut, can: (p) => !!user?.permissions.includes(p) }),
    [user, ready, signIn, signOut],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}

export const ROLE_LABEL: Record<string, string> = {
  triage_nurse: "Triage Nurse", charge_nurse: "Charge Nurse", physician: "Physician", admin: "Administrator", viewer: "Display only",
};
