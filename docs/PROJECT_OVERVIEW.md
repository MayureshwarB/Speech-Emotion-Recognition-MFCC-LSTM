# Project Overview

## Problem
Text alone does not capture all information carried by speech. This project explores assigning an emotion label from vocal features in acted speech recordings.

## Goal
Use 40 MFCC audio features and an LSTM-based classifier to select one of seven emotion categories, and expose inference through a simple WAV-upload interface.

## Pipeline
WAV audio → Librosa preprocessing → 40 MFCC coefficients averaged over time → reshape to `(40, 1)` → LSTM/dense classifier → seven output scores → top predicted label.

## Data and split
The model metadata records 2,800 TESS WAV clips: 2,240 training samples and 560 evaluation samples, split 80/20.

## Recorded metrics
- Training accuracy: 99.64%
- Evaluation accuracy: 98.39%
- Training–evaluation gap: 1.25 percentage points
- Macro F1-score: 0.984

The evaluation partition is also used as validation during training, so the metrics are not an independent, untouched benchmark. See `models/evaluation_metrics.json` for per-class metrics.

## Label mapping
The saved model outputs labels in this order: `angry`, `disgust`, `fear`, `happy`, `neutral`, `ps`, `sad`. The class abbreviation `ps` is displayed as “Pleasant Surprise,” as recorded in the model metadata.

## Limitations
The model was trained on one acted-speech dataset and may generalize poorly to spontaneous speech or new speakers and recording conditions. Its output is not a psychological diagnosis or objective measurement of a person's inner emotion.
