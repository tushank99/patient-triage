# Speech-to-text benchmark (faster-whisper, int8 CPU)

16 clinical complaints x 3 Windows TTS voices, clean and with white noise at 10 dB SNR.

| model / vocabulary / audio | WER | medical-term recall | symptom F1 | duration | pain score | errors flagged for review | error-free but flagged | real-time factor |
|---|---|---|---|---|---|---|---|---|
| tiny.en | plain | clean | 0.123 | 0.71 | 0.88 | 0.94 | 0.94 | 0.60 | 0.50 | 0.22 |
| tiny.en | plain | noise 10 dB SNR | 0.179 | 0.68 | 0.73 | 0.94 | 1.00 | 0.92 | 1.00 | 0.23 |
| tiny.en | medical vocab | clean | 0.082 | 1.00 | 0.88 | 0.94 | 1.00 | 0.71 | 0.67 | 0.53 |
| tiny.en | medical vocab | noise 10 dB SNR | 0.164 | 0.93 | 0.83 | 0.94 | 0.94 | 1.00 | 1.00 | 0.51 |
| base.en | plain | clean | 0.103 | 0.82 | 0.88 | 0.94 | 1.00 | 0.50 | 0.50 | 0.40 |
| base.en | plain | noise 10 dB SNR | 0.149 | 0.75 | 0.73 | 0.94 | 1.00 | 0.83 | 0.50 | 0.29 |
| base.en | medical vocab | clean | 0.262 | 0.96 | 0.88 | 0.94 | 0.94 | 0.92 | 1.00 | 1.34 |
| base.en | medical vocab | noise 10 dB SNR | 0.179 | 0.93 | 0.79 | 0.94 | 1.00 | 1.00 | 1.00 | 0.93 |
