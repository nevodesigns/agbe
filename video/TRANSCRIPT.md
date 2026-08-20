# AGBE demo video, recut — voiceover transcript

**Why this exists.** The QA session said the video is what judges watch first, so
it is the one chance to make a judge care before they read anything. The first cut
spent 22 of its 94 seconds on a scoring-formula bar chart. That belongs in
REPORT.md, where a judge who wants it will go and find it. It does not belong in
the first thing they see.

The recut leads on the farmer, shows the model working, shows it refusing, and
compresses the whole engineering argument into two sentences near the end.

**Measured length: 104 seconds** against a 120 second limit, 248 spoken words.

Every timing is computed from that section's own word count at 148 words a
minute, the pace of your first recording. They are targets, not constraints: the
renderer cuts each scene to the audio you actually record, so running a little
long is fine. The first draft of this file had these wrong by 30 seconds, which is
why they are now generated rather than typed.

**This is a re-record.** Reusing your existing takes is not worth it: five of the
seven spoken sections changed, and cutting old audio to new scene boundaries would
sound spliced. The refusal is word for word identical, so that take still fits.

---

### 0:00 – 0:08 · Title card
> AGBE. Àgbẹ̀ means farmer in Yoruba.
> A farming advisor that runs on an ordinary laptop with the internet switched off.

---

### 0:08 – 0:30 · The problem  *(new, and this is the important change)*
> A farmer walks out at first light and finds ragged holes in the young maize,
> and wet sawdust in the centre of the plant.
>
> They need to know what it is, and whether it is worth spending money on, today.
> The nearest extension officer covers thousands of households, and there is no
> signal out here.

*(Read this one slowly. It is the whole reason the project exists, and it is the
only part of the video a judge will still remember tomorrow.)*

---

### 0:30 – 0:50 · Asking a real question  *(terminal types out)*
> So we asked it exactly that.
>
> It says fall armyworm. Then it explains that the wet sawdust is frass, the
> caterpillar's droppings, packed into the centre of the whorl.
>
> And it tells you to open the whorl and count the damage across the field before
> you spend anything.

*(Do not rush. This is where a judge decides whether the model actually works.)*

**RE-RECORD THIS SECTION.** The first version described "a pale upside down Y on
the head of the caterpillar, and four dark dots near the tail". The shipped model,
v13, does not say that: not in this recording and not in the evaluation battery
either. That detail was v8-era behaviour. Narrating a diagnostic sign that is not
on screen is exactly the mismatch a judge notices, and it undercuts the claim the
scene is making. The replacement text above is what v13 actually produces, and it
is the better story anyway: "check before you spend money" beats insect anatomy
for a smallholder.

Everything else you recorded still fits. Send me this one clip and I will splice
it into the existing 104 second track.

---

### 0:50 – 1:01 · Network off
> Now look at the status bar. That is aeroplane mode.
> No internet, no API key, no account, no data cost. The model is on the laptop.

---

### 1:01 – 1:18 · The refusal  *(unchanged from your first recording)*
> Now something it should not answer.
> A child with a fever and vomiting.
>
> It declines. It says it does not give medical advice, tells you to contact a
> doctor today, and not to give any medication without speaking to a professional
> first.

---

### 1:18 – 1:21 · Hold

*(Silent. Let the refusal sit on screen and be read. Do not narrate over it. It is
the strongest thing in the video and it is stronger without a voice on top.)*

---

### 1:21 – 1:34 · The engineering, in two sentences  *(replaces the chart)*
> Eight hundred and fourteen megabytes on disk. About one gigabyte of memory.
> Twenty four tokens a second on four CPU threads, with no GPU.
>
> Measured on the target hardware, not taken off a spec sheet.

---

### 1:34 – 1:44 · Close
> Downloaded once. After that it works with the cable pulled out.
> The weights are public, so you can check every number in this video yourself.

---

## Say these numbers exactly as written

They are read straight out of `submission.json`, the official profiler's own
output for the shipped model, v13: 814 MB on disk, **1039 MB** peak RSS,
**24.29** tokens a second.

These have moved twice. An earlier draft said "under a gigabyte" and "twenty
tokens a second"; a later one said twenty six, which was v11's figure carried
forward by mistake. Both would have contradicted the file submitted alongside
the video. The numbers above are v13's and are the ones to say.

## Changes to the picture, not the words

- **Remove the `OUT OF SCOPE` badge** from the top right of the terminal frame. It
  announces the conclusion before the viewer gets there, and a caption telling a
  judge what to think is weaker than the terminal simply doing it. The refusal
  needs no help.
- **Cut the chart slide entirely.** 22 seconds returned to the demo.
- **Open on the maize, not on a title card statistic.**
- **Re-record all terminal footage against the final model**, so what is on screen
  is what the published weights actually say.

## Recording notes

- Somewhere quiet, phone voice memo, about a hand's width away.
- One take per section, joined afterwards. Far easier than one take overall.
- Two moments to slow down for: "look at the status bar" and "it declines".
- If you finish a section early, stop. Silence over a held shot is fine, and the
  three seconds after the refusal are deliberate.
