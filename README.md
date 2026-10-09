# Speech Emotion Recognition using MFCC and LSTM

A deep-learning project that classifies the emotion label associated with a short speech recording. It uses **Mel-Frequency Cepstral Coefficients (MFCCs)** as audio features and an **LSTM neural network** for seven-class classification. A Streamlit app lets a user upload a WAV file and view the model's predicted label and scores.

> **Important limitation:** This is an educational/research demo using acted speech from TESS. Voice emotion is subjective and context-dependent. The model output is not a clinical or psychological assessment, and its scores are not calibrated confidence values.

## Try the application

The Streamlit interface is included in `app/streamlit_app.py`. To run it locally, follow the installation steps below. A public hosted demo is not configured yet.

## Results from the saved model run

The metrics below are read from `models/evaluation_metrics.json`, uploaded with the trained model artifact.

| Metric | Recorded value |
|---|---:|
| Dataset clips | 2,800 |
| Training samples | 2,240 (80%) |
| Evaluation samples | 560 (20%) |
| Training accuracy | 99.64% |
| Evaluation accuracy | 98.39% |
| Training–evaluation gap | 1.25 percentage points |
| Macro F1-score | 0.984 |

## How it works

1. Load a short WAV recording.
2. Use Librosa to load up to three seconds of audio, beginning at a 0.5-second offset.
3. Extract 40 MFCC coefficients and average them across time.
4. Reshape the 40-value feature vector for the model input `(40, 1)`.
5. Pass the input through an LSTM and dense layers.
6. Return seven model output scores and display the highest-scoring emotion label.

## Model architecture

- Input shape: `(40, 1)`
- LSTM: 128 units
- Dense: 64 units, ReLU
- Dropout: 20%
- Dense: 32 units, ReLU
- Dropout: 20%
- Output: 7 units, Softmax
- Optimizer used for training: Adam
- Loss used for training: categorical cross-entropy

## Emotion labels

The model's output order is stored in `models/class_names.json`:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `ps`, `sad`

The dataset abbreviation `ps` means **pleasant surprise**. The app uses “Pleasant Surprise” as the display name.

## Repository structure

```text
speech-emotion-recognition/
├── app/
│   └── streamlit_app.py
├── docs/
│   └── PROJECT_OVERVIEW.md
├── models/
│   ├── speech_emotion_lstm.keras
│   ├── class_names.json
│   └── evaluation_metrics.json
├── notebooks/
│   ├── speech_emotion_recognition.ipynb
│   └── train_model_on_colab.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Run locally

Python 3.10 or 3.11 is recommended. In a terminal opened at the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the requirements and start the app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit, upload a short WAV recording, and click **Predict emotion**.

## Dataset and attribution

The project uses the Toronto Emotional Speech Set (TESS), described by Pichora-Fuller and Dupuis (2020):

> Pichora-Fuller, M. Kathleen and Kate Dupuis (2020). *Toronto emotional speech set (TESS)*. Borealis. https://doi.org/10.5683/SP2/E8H2MF

The dataset is distributed under **CC BY-NC 4.0**. This repository does not include the audio dataset. Review the dataset license before commercial use and include required attribution when redistributing related materials.

## Limitations and future improvements

- Add a speaker-independent evaluation and test with an independent dataset.
- Compare the current mean-MFCC representation with a sequence-preserving MFCC input.
- Test different microphones, accents, and real-world recording conditions.
- Add optional microphone recording only with explicit user consent.
- Treat predictions cautiously; vocal emotion labels are not objective measurements of a person's internal state.

## Project source

This repository packages the Group 15 case study, “Deep Learning Case Study on Speech Emotion Recognition Using MFCC & LSTM,” guided by Prof. Rasika Kulkarni. Preserve applicable course, team, and dataset attribution when sharing the work.
