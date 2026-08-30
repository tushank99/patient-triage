/* eslint-disable prettier/prettier */
import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Building2, ActivitySquare } from "lucide-react";

export function SettingsModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [tier, setTier] = React.useState("LEVEL_1_TRAUMA");
  const [ctStatus, setCtStatus] = React.useState("ONLINE");

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[450px]">
        <DialogHeader>
          <DialogTitle>Platform Settings</DialogTitle>
        </DialogHeader>
        <div className="py-4 space-y-6">
          <div className="space-y-4">
            <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
              <Building2 className="w-4 h-4" /> Hospital Scale & Tier
            </h4>
            <div className="space-y-2">
              <label className="flex items-start gap-3 p-3 border rounded-lg cursor-pointer hover:bg-slate-50">
                <input 
                  type="radio" 
                  name="tier" 
                  value="LEVEL_1_TRAUMA" 
                  checked={tier === "LEVEL_1_TRAUMA"} 
                  onChange={(e) => setTier(e.target.value)} 
                  className="mt-1"
                />
                <div>
                  <p className="font-semibold text-sm">Level 1 Trauma Center (Urban)</p>
                  <p className="text-xs text-muted-foreground">High-acuity, full specialist availability, massive scale.</p>
                </div>
              </label>
              <label className="flex items-start gap-3 p-3 border rounded-lg cursor-pointer hover:bg-slate-50">
                <input 
                  type="radio" 
                  name="tier" 
                  value="RURAL_CLINIC" 
                  checked={tier === "RURAL_CLINIC"} 
                  onChange={(e) => setTier(e.target.value)} 
                  className="mt-1"
                />
                <div>
                  <p className="font-semibold text-sm">Rural Clinic (Low Resource)</p>
                  <p className="text-xs text-muted-foreground">Limited staff, AI scales up to catch missed critical symptoms.</p>
                </div>
              </label>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
              <ActivitySquare className="w-4 h-4" /> Equipment Status
            </h4>
            <div className="space-y-2">
              <select 
                value={ctStatus} 
                onChange={(e) => setCtStatus(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm"
              >
                <option value="ONLINE">CT Scanner: ONLINE</option>
                <option value="OFFLINE">CT Scanner: OFFLINE</option>
                <option value="MAINTENANCE">CT Scanner: MAINTENANCE</option>
              </select>
              {ctStatus === "OFFLINE" && (
                <p className="text-xs text-red-600 font-medium pt-1">
                  AI will adjust stroke/trauma triage recommendations to account for lack of imaging.
                </p>
              )}
            </div>
          </div>
        </div>
        <div className="flex justify-end pt-4 border-t">
          <button 
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-bold hover:bg-blue-700"
          >
            Save Settings
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
