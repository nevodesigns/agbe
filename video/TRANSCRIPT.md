# AGBE demo video — voiceover transcript

**Video:** `agbe-demo.mp4`, 1920x1080, silent, **110 seconds** (limit is 120).
**Your job:** read this over the video. The timings are scene boundaries, so if
you land near them the words sit on the right picture.

Read it plainly. No hype, no "revolutionary". The material is strong enough
stated flatly, and judges hear a lot of hype.

About 250 spoken words, comfortable at a natural pace across 110 seconds. If you
finish a section early, pause. Silence over a held shot is fine.

---

### 0:00 – 0:08 · Title card

> AGBE. Àgbẹ̀ means farmer in Yoruba.
> It is a farming advisor that runs on an ordinary laptop, with the internet
> switched off.

---

### 0:08 – 0:18 · The problem

> Nigeria has about one agricultural extension officer for every few thousand
> farming households.
> The knowledge that would raise a smallholder's yield is not secret and it is not
> new. It sits in extension manuals. It just never reaches the field.

---

### 0:18 – 0:38 · Asking a real question (terminal types out)

> So we asked it what farmers actually ask.
> Holes in the young maize leaves, and something like wet sawdust in the centre of
> the plant.
>
> It says fall armyworm. Then it tells you how to check before you spend any
> money: a pale upside down Y on the head of the caterpillar, four dark dots near
> the tail.
>
> That is real fall armyworm biology, not a guess.

*(Pause and let the text finish appearing.)*

---

### 0:38 – 0:50 · Network off

> This is the part that matters.
> There is no internet. No API key, no account, no data cost.
> The model is on the laptop. It answers the same with the cable pulled out.

---

### 0:50 – 1:10 · The refusal

> Now something it should not answer.
> A child with a fever and vomiting.
>
> It declines. It says it does not give medical advice, tells you to contact a
> doctor today, and not to give any medication without speaking to a professional
> first.
>
> Getting a one billion parameter model to say no took four attempts. For a tool
> used by people with no alternative, knowing where it stops matters as much as
> what it knows.

---

### 1:10 – 1:26 · The chart

> The challenge publishes its scoring formula, so we read it before writing code.
> Throughput above fifteen tokens a second earns nothing, and memory is charged
> linearly.
>
> That makes the obvious move, running the biggest model that fits in eight
> gigabytes, exactly backwards. The three billion model gives up fourteen and a
> half points before answering a single question.
>
> So we measured five candidates, and built the one billion.

---

### 1:26 – 1:40 · The numbers

> Eight hundred and fourteen megabytes on disk. Under a gigabyte of memory.
> Twenty four tokens a second on four CPU threads with no GPU.
>
> Forty seven and a half of the fifty available engineering points, measured on
> the target hardware rather than estimated.

---

### 1:40 – 1:50 · Close

> Downloaded once. After that it works with the cable pulled out.
> The weights are public, so you can check every number in this video yourself.

---

## Recording notes

- **Record somewhere quiet.** A phone voice memo held about a hand's width away
  is fine and beats a laptop microphone.
- **One take per section** is far easier than one take overall. Join them after.
- **Do not rush 0:18 to 0:38.** That is where a judge decides whether the model
  actually works, and the text is appearing on screen at the same time.
- **The two strongest moments** are "there is no internet" at 0:38 and "it
  declines" at 0:50. Slow down slightly for both.

## Putting the voice on the video

Once the audio is a single file, say `voice.m4a`:

```bash
ffmpeg -i video/agbe-demo.mp4 -i voice.m4a \
  -c:v copy -c:a aac -shortest video/agbe-demo-final.mp4
```

Send me the audio and I will run that and check the sync, or run it yourself.
Then upload to YouTube as **Unlisted** and paste the link into the Devpost
"Video demo link" field.
