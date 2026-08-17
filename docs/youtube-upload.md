# YouTube upload sheet — AGBE demo

Everything to paste when uploading. Video file: `agbe-demo-captioned.mp4`
(or `agbe-demo.mp4` plus `agbe-demo.srt` as a caption track).

---

## Title

Pick one. The first is strongest: it leads with the surprising claim rather than
the project name, which nobody has heard of yet.

```
AGBE — a farming advisor that runs offline on an 8GB laptop
```

```
I trained a 1B model to diagnose crop pests offline. No internet, no API key.
```

```
AGBE: offline AI farm advisor for African smallholders (Gemma 3 1B, llama.cpp)
```

Keep it under about 60 characters so it does not truncate in search results.

---

## Description

Paste this whole block. The first two lines are what shows before "read more",
so they carry the pitch.

```
AGBE is a farming advisor that runs entirely on an ordinary 8GB laptop with the
internet switched off. No cloud, no API key, no account, no data cost.

Nigeria has roughly one agricultural extension officer for every few thousand
farming households. The advice that would raise a smallholder's yield already
exists in extension manuals. It just never reaches the field, because the last
mile has no officer and often no signal.

Built for the Africa Deep Tech Challenge 2026 (agriculture track).

CHAPTERS
0:00  What AGBE is
0:07  The problem
0:16  Asking it a real question: fall armyworm in maize
0:30  The same answer with aeroplane mode on
0:39  Asking something it should refuse
0:57  Why a 1B model beats the biggest one that fits
1:12  The measured numbers
1:24  Where to get it

EVERY TERMINAL SCENE IS REAL
The model was run in a live terminal and recorded with timestamps, so the pause
before the first token and the speed of generation are the model's own. Nothing
is sped up.

THE NUMBERS, MEASURED NOT ESTIMATED
Model size      814 MB (Gemma 3 1B, LoRA r32, GGUF Q4_K_M)
Memory at peak  0.88 GB
Speed           20 to 27 tokens/sec on 4 CPU threads, no GPU
Score           47.5 of the 50 available engineering points

WHY A 1B AND NOT A 3B
The challenge publishes its scoring formula. Throughput above 15 tokens/sec
earns nothing, and memory is charged linearly, so running the largest model that
fits in 8GB is exactly backwards. We measured five candidates on the target
hardware. The 3B gives up 14.5 points before answering a single question.

WHAT IT REFUSES
It will not give an agrochemical dose, because rates differ by product and a
confident wrong number is dangerous. It will not quote a market price. Asked
about a child with a fever it declines and points to a clinic.

TRY IT YOURSELF
The weights are public, so you can check every number in this video.

  curl -L -o agbe.gguf https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf
  llama-cli -m agbe.gguf -t 4 -ngl 0 -c 2048 -st -p "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"

LINKS
Site and build notes  https://agbe-farm.vercel.app
Source and corpus     https://github.com/nevodesigns/agbe
Weights               https://huggingface.co/NEVODESIGN/agbe-1b

Base model Gemma 3 1B, used under the Gemma Terms of Use. Runs on llama.cpp.
Built by Nwokolo Victor Oluebubechukwu, Lagos.

#OfflineAI #EdgeAI #AfricaTech #AgriTech #SmallLanguageModels
```

---

## Hashtags

YouTube shows the **first three** above the video title, so order matters. These
three go at the very end of the description:

```
#OfflineAI #EdgeAI #AfricaTech
```

The wider set, for the description and any social posts:

```
#OfflineAI #EdgeAI #AfricaTech #AgriTech #SmallLanguageModels #Gemma #llamacpp
#OnDeviceAI #AIforAgriculture #Nigeria #MachineLearning #OpenSource
#AfricaDeepTech #ADTC2026 #LocalLLM
```

**Do not exceed 15 hashtags.** Over that, YouTube ignores all of them, which is
a silent failure most people never notice.

---

## Tags (the separate Tags field, not hashtags)

Paste comma separated:

```
offline AI, edge AI, on-device AI, small language model, SLM, Gemma 3, llama.cpp,
GGUF, quantization, LoRA fine-tuning, agriculture AI, agritech, Africa tech,
Nigeria, smallholder farming, fall armyworm, extension services, local LLM,
CPU inference, no internet AI, Africa Deep Tech Challenge, machine learning,
open source AI, low resource computing
```

---

## Settings

| Field | Value |
|---|---|
| Visibility | **Unlisted** is enough for Devpost. Public if you want reach. |
| Category | Science & Technology |
| Language | English |
| Captions | Upload `agbe-demo.srt` under Subtitles, English |
| Made for kids | **No** |
| Comments | On |
| Chapters | Automatic, because the description has timestamps starting at 0:00 |

**Do not skip the caption upload.** Even with captions burned in, the SRT makes
the video searchable inside YouTube and readable by screen readers.

---

## Thumbnail

Grab the frame at about **0:34**: aeroplane mode visible in the status bar, the
fall armyworm answer on screen. That single image carries the whole idea, a real
terminal answering with no network.

```bash
ffmpeg -ss 34 -i agbe-demo-captioned.mp4 -frames:v 1 thumbnail.png
```

Add three or four words in large text if you want, no more:

```
NO INTERNET. STILL ANSWERS.
```

---

## Pinned comment

Worth pinning, because it invites verification, which is the strongest thing this
project has:

```
The weights are public and everything in this video is reproducible:
https://huggingface.co/NEVODESIGN/agbe-1b

Full build notes, including the model versions that failed and why:
https://agbe-farm.vercel.app/notes/
```
