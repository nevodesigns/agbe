# Corpus specification

What we are optimising for, stated precisely, because every later decision refers back
to this.

## The scoring reality

`S_acc` is 50% of the total and is **judged by humans reading generated answers**. We
submit 2 test prompts; organisers add 2 hidden prompts in the agriculture domain. There
is no official dataset and no automated accuracy metric in the scoring path.

Three consequences:

1. **Breadth beats depth.** We cannot predict the 2 hidden prompts. A model that answers
   any reasonable smallholder question competently scores better than one that is
   excellent on cassava and lost on poultry.
2. **Style is scored, not just facts.** A judge reading a 1B model's answer is grading
   usefulness. Structure, specificity and honest hedging read as competence.
3. **Overfitting is explicitly tested.** The hidden prompts exist to catch models tuned
   to their own submitted prompts. So we train general agronomic reasoning patterns, not
   memorised pairs.

## The reader we are writing for

A Nigerian smallholder farmer or a agricultural extension officer, on a $400 laptop, with
no internet. They need an answer they can act on this week, using inputs they can
actually buy locally.

This is also the African Use Case claim (`african_alpha_claim: true`, worth $1,500 and up
to 10 bonus points), so the local grounding is not decoration. It is the submission's
argument.

## Answer style contract

Every training response follows this, so the model learns one voice:

- **Lead with the answer.** No preamble, no restating the question.
- **Be specific and local.** Name varieties, months, spacings, rates. "Plant at the start
  of the rains" is weak. "Plant maize when rains are established, usually late April in
  the middle belt, at 75cm between rows and 25cm within rows" is what gets scored.
- **Structure when there are steps.** Short numbered or dashed lists. Prose for
  explanations.
- **Use local units and names.** Hectares and kilograms, plus local crop names where they
  are what people say (egusi, ugu, tatase).
- **Hedge honestly where it matters.** Agrochemical rates vary by product, so point at
  the label and the local extension officer rather than inventing a dose. A model that
  says "follow the rate on the label, and confirm with your extension officer" is scored
  as trustworthy. A model that invents millilitres per litre is scored as dangerous.
- **Length: 60 to 160 words, measured rather than targeted.** The first version of this
  document asked for 80 to 220 and never checked. Measured on the trained model the
  median answer is about 75 words, and 64% of the corpus sat under the old floor, so the
  contract described something the data never did. The band above is what the corpus and
  the model actually produce. It is also where they behave best: the failure mode of a 1B
  is drift at the tail, and every attempt to pad answers toward 200 words bought
  fabrication rather than substance. A correct 75-word answer scores better than a
  120-word answer with an invented sentence at the end.
- **Never invent a statistic.** No fake yield figures, no fake prices. Prices and yields
  vary, so describe the direction and the factors instead.

## Safety boundary

Agriculture touches pesticides and veterinary treatment. The rule is: name the active
ingredient class and the practice, defer the dose to the product label and the extension
officer. Never give a human medical instruction. This protects the score, because the
judging panel includes agronomists who will notice invented chemistry.

## Language scope

`language_scope: ["en"]`

English only. Nigerian Pidgin (`pcm`) was built, trained and tested across several
model builds, then **withdrawn**. It answered correctly in v5 and in v6 named
*amala*, a food, as a maize pest. A capability that works one time in two is not a
capability, so the claim was removed rather than shipped and hoped for.

Pidgin examples remain in the corpus because they still teach the model to
recognise how Nigerian farmers describe symptoms, which improves English answers
to those phrasings. They are simply no longer claimed as a supported output
language.

The African Use Case claim is unaffected: it rests on the domain being cassava
mosaic, striga, aflatoxin, Newcastle disease and harmattan planting windows, not
on language.

## Anti-overfitting design

- No training pair may duplicate either of our 2 submitted test prompts.
- Each topic appears in several question *forms* (what, why, how, when, troubleshooting,
  comparison, cost) so the model generalises across phrasing.
- A held-out slice is kept unseen for our own qualitative checks, standing in for the
  judges' hidden prompts.

## Composition targets

| Slice | Share | Purpose |
| --- | --- | --- |
| Crop agronomy | 34% | The core of the domain |
| Pests and diseases | 20% | The most common real question |
| Livestock, poultry, aquaculture | 14% | Breadth against hidden prompts |
| Soil, fertiliser, water | 10% | Underpins everything else |
| Post-harvest, storage, market | 9% | Where smallholders lose most income |
| Nigerian Pidgin pairs | 6% | Trains recognition of farmer phrasing; not a claimed output language |
| General instruction replay | 7% | Guards against catastrophic forgetting |

The replay slice matters. LoRA on a narrow domain will damage general instruction
following, and a judge whose hidden prompt is phrased conversationally will notice. We
mix in general data to hold the base model's manners.
