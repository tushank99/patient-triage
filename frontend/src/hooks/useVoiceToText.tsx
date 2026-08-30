/* eslint-disable prettier/prettier */
import * as React from "react";

interface SpeechRecognitionAlternative {
  transcript: string;
}
interface SpeechRecognitionResult {
  0: SpeechRecognitionAlternative;
  length: number;
}
interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionResult;
  length: number;
}
interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}
interface ISpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: () => void;
  onresult: (event: SpeechRecognitionEvent) => void;
  onerror: (event: unknown) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
}
interface IWindow extends Window {
  SpeechRecognition?: new () => ISpeechRecognition;
  webkitSpeechRecognition?: new () => ISpeechRecognition;
}

export function useVoiceToText() {
  const [isListening, setIsListening] = React.useState<boolean>(false);
  const [transcript, setTranscript] = React.useState<string>("");
  const [error, setError] = React.useState<string | null>(null);
  
  const recognitionRef = React.useRef<ISpeechRecognition | null>(null);

  const startListening = React.useCallback(() => {
    const win = window as unknown as IWindow;
    const SpeechRecognition = win.SpeechRecognition || win.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError("Voice input not supported in this browser. Please type.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;

  
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      
      const finalTranscript = event.results[0]?.[0]?.transcript || "";
      setTranscript(finalTranscript);
    };

    recognition.onerror = () => {
      setError("Microphone disconnected or network error. Reverting to manual entry.");
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  }, []);

  const stopListening = React.useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsListening(false);
  }, []);

  return {
    isListening,
    transcript,
    startListening,
    stopListening,
    error,
    setTranscript,
  };
}