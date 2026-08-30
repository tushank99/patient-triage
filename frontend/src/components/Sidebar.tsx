/* eslint-disable prettier/prettier */
import { Link } from "@tanstack/react-router";
import { useState } from "react";
import {
  ClipboardList,
  LayoutDashboard,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Stethoscope,
} from "lucide-react";
import { SettingsModal } from "./SettingsModal";

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const links = [
    { label: "Dashboard", icon: LayoutDashboard, to: "/" as const },
    { label: "Patient Details", icon: ClipboardList, to: "/patients" as const },
  ];

  return (
    <>
      <aside className={`fixed inset-y-0 left-0 z-30 flex flex-col border-r border-border bg-card transition-[width] duration-200 ${collapsed ? "w-20" : "w-64"}`}>
        <div className={`flex items-center border-b border-border py-5 ${collapsed ? "justify-center px-3" : "gap-3 px-6"}`}>
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <Stethoscope className="h-5 w-5" />
          </div>
          {!collapsed && (
            <div className="min-w-0 flex-1">
              <p className="truncate text-base font-bold leading-tight tracking-tight">TriageFlow</p>
              <p className="truncate text-xs text-muted-foreground">Emergency Department</p>
            </div>
          )}
          <button
            type="button"
            onClick={onToggle}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={`shrink-0 text-muted-foreground transition-colors hover:text-foreground ${collapsed ? "absolute right-2 top-3" : ""}`}
          >
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </button>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-4">
          {links.map(({ label, icon: Icon, to }) => (
            <Link
              key={label}
              to={to}
              aria-label={label}
              title={collapsed ? label : undefined}
              activeProps={{
                className: "bg-accent text-accent-foreground shadow-sm ring-1 ring-border"
              }}
              inactiveProps={{
                className: "text-muted-foreground hover:bg-muted hover:text-foreground"
              }}
              className={`flex w-full items-center rounded-lg py-2.5 text-sm font-medium transition-colors ${collapsed ? "justify-center px-2" : "gap-3 px-3"}`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </Link>
          ))}
          {/* Settings button */}
          <button
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            aria-label="Settings"
            title={collapsed ? "Settings" : undefined}
            className={`flex w-full items-center rounded-lg py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground ${collapsed ? "justify-center px-2" : "gap-3 px-3"}`}
          >
            <Settings className="h-4 w-4 shrink-0" />
            {!collapsed && "Settings"}
          </button>
        </nav>

        <div className="border-t border-border p-3">
          <button
            type="button"
            aria-label="Logout"
            title={collapsed ? "Logout" : undefined}
            className={`flex w-full items-center rounded-lg py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-destructive ${collapsed ? "justify-center px-2" : "gap-3 px-3"}`}
          >
            <LogOut className="h-4 w-4 shrink-0" />
            {!collapsed && "Logout"}
          </button>
        </div>
      </aside>
      
      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </>
  );
}
