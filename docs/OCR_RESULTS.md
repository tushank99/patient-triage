# OCR benchmark (rapidocr-ppocr-onnx)

12 synthetic documents (lab reports with tables, medication/allergy lists) x 6 degradations. Field threshold 0.85.

| degradation | CER | field accuracy | labs | dates | MRN | meds | allergies | auto-accepted | wrong among auto-accepted | errors held for review |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 0.190 | 0.81 | 0.97 | 0.25 | 1.00 | 0.71 | 1.00 | 0.68 | 0.02 | 0.94 |
| skew 2.5deg | 0.195 | 0.66 | 0.80 | 0.25 | 1.00 | 0.33 | 1.00 | 0.60 | 0.07 | 0.87 |
| noise+blur | 0.215 | 0.67 | 0.93 | 0.25 | 1.00 | 0.21 | 1.00 | 0.60 | 0.02 | 0.97 |
| low contrast | 0.175 | 0.79 | 0.90 | 0.25 | 1.00 | 0.71 | 1.00 | 0.68 | 0.02 | 0.95 |
| jpeg q15 + downscale | 0.243 | 0.50 | 0.57 | 0.25 | 0.92 | 0.12 | 0.92 | 0.39 | 0.06 | 0.96 |
| rotated 90deg | 0.189 | 0.81 | 0.93 | 0.25 | 1.00 | 0.75 | 1.00 | 0.69 | 0.02 | 0.94 |
