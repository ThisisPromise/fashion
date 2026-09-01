# Recruiter Critique &amp; Builder's Defense

Two voices. First the recruiter exposes every concern raised during the
walkthrough; then the builder answers each one and defends the decisions.

---

# PART A — THE RECRUITER'S CRITIQUE

I read every file in this submission. It is a strong piece of work overall —
transfer learning, a geometric placement guarantee, and graceful degradation
are genuinely good instincts. But a strong interview candidate is judged on
the *hard* parts, and there are real gaps, questionable choices, and an
honesty problem I want resolved. Here is my honest, critical read.

## A1. The `NOTE.md` that doesn't exist

The README explicitly promises documentation of "splits, metric, and known
failure cases" in `NOTE.md` (README.md:21 and :101). There is no `NOTE.md`
anywhere in the folder. Either you forgot to include a deliverable, or your
documentation is aspirational rather than real.

This matters to me for two reasons:
1. It's the difference between "known failure cases are documented" and
   "there are known failure cases." If they exist, you should want to show
   them. If they don't exist, you misled the reader.
2. A submission that overstates its own documentation is a red flag about
   how the rest of the claims should be audited.

**My call:** this gets resolved first. Show me the actual `NOTE.md`, or fix
the references. As-is it reads like the README was written for a version of
the project that no longer exists.

## A2. Sleeve IoU of 0.473 is the elephant in the room

Your headline metric hides a real deficiency. Background (0.959) and body
(0.647) are okay, but the model can barely separate sleeves from the body.
Mean IoU 0.693 is flattered by how easy background is.

The bigger concern is *honesty about impact*: half of your evaluation cases
are "chest" placements, which sit on the body class that scores decently. So
your evaluation set is engineered around your model's strongest class, and
the weak class — sleeve — is barely exercised as a *placement constraint*.
Sleeve quality directly determines whether artwork is refused for risk of
crossing a seam, and that is the very thing your system claims to prevent.

**What I want to hear:** why sleeves are hard, what you tried, and whether
you have numbers on how often sleeve error actually caused a bad placement
or a wrong refusal.

## A3. The flat-lay "fallback" may be papering over a real failure

You detected that your model was trained on worn garments and mislabels
flat lays. Your response was a geometric heuristic that ignores the model
entirely on those images. I have three concerns:

1. **You never retrained or fine-tuned on flat-lay data.** You chose to paper
   over the distribution gap with a hand-tuned `0.28` margin fraction
   instead of fixing the model. Why?
2. **The heuristic is a hidden assumption.** You assume every flat-lay
   garment's sleeves splay symmetrically to the outer ~28% of the bounding
   box. That's false for some garments — a cardigan or a coat laid with arms
   down, a draped scarf, a posed flat lay. When it's wrong, you have no
   signal to detect it.
3. **The sleeve/body precedence may be hiding errors.** Sleeve overrides body
   `prepare_data.py:54`. For the shoulder region especially, that's a choice
   that makes evaluation look cleaner while potentially mislabeling a
   boundary that placement depends on.

## A4. The "guarantee" is narrower than you claim

Your inscribed-circle anchor is elegant, but the word "guarantee" (
`placement.py:144`, `:213`) is doing a lot of work. What is actually bounded
is: *if the segmentation were perfect*, the artwork fits inside the target
zone. Your mask is not perfect (see sleeve IoU). So the real-world guarantee
is only as good as the segmentation. I'm not saying the geometry is wrong —
I'm saying the way it's framed risks overstating what was solved.

## A5. Determinism is real but fragile and slow

You correctly claim the placement is deterministic (`placement.py:168-170`).
Two caveats:

1. **CPU-only.** `train.py:106` hardcodes `torch.device("cpu")`, so
   determinism is only guaranteed in a single-threaded-ish CPU environment.
   As soon as anyone runs this on GPU, or with batched/parallel inference or
   cuDNN nondeterminism, your claim needs re-examining. You never define the
   scope of the determinism promise.
2. **Speed.** An ImageNet backbone + FPN decoder + upsampling + MediaPipe
   pose, all on CPU, for every garment photo. Fine for a demo. If this is a
   production "preview tool," per-image latency matters and you've said
   nothing about it.

## A6. Brittle instruction parsing

`parse_instruction` is pure substring matching (`placement.py:72-83`).
"front, left chest" works. But so would "front, left" (interpreted as left
chest even though no chest word was present — no, wait, `vertical` falls
back to "centre", so it becomes left-centre, which is a *silently different
placement than a human asked for*). Unrecognized phrases fall back to
"centre" silently. Is a wrong-but-successful placement worse than a refusal?
For a tool whose whole selling point is "refuse rather than guess," silently
guessing a different region on ambiguous input contradicts the philosophy.

## A7. No tests, no errors for missing models

There is not a single test in the repository. The README gives run commands
but no test command. For code with non-trivial geometry (distance transform,
mask precedence, compositing), I would expect *some* unit tests verifying
e.g. that the anchor is always inside the region, or that rejection fires on
an empty mask. The model file missing (`pose_landmarker_lite.task`) degrades
silently to `None` — which is graceful for placement, but it also means a
broken install gives you *confident-looking* flat-lay heuristics with no
warning.

## A8. Reproducibility / environment fragility

`requirements.txt` pins exact versions. Good. But it lists both
`opencv-contrib-python` and `opencv-python-headless` side by side — a
known-source of runtime conflicts on some platforms. `sounddevice` is in
there for a task that has nothing to do with audio. And the mediapipe model
file is fetched from a URL in a README rather than vendored. None of this is
fatal, but it says the dependency story wasn't fully thought through.

## A9. Evaluation is thin and somewhat circular

Eight case rows in `evaluation_cases.csv` is not a real evaluation harness.
The cases overlap the training distribution (several are `data/processed`
images), so placing on them is somewhat self-referential — the model is
being asked to place on images from the same set it was trained on. The
`sample_garments/suit.jpg` cases are the only genuinely out-of-distribution
tests, and they lean on the fallback. A production placement tool needs to
measure *placement outcome* (did it land where a human would put it, was a
bad placement refused) against a labeled set, not just whether the pipeline
ran without crashing.

## A10. Data-engineering leakage risk

The enrichment pipeline is intentional and clean, but confirm: the class
weights are computed from the *train* split only (`train.py:26-31`) — good.
But the split itself was made *after* rasterizing and saving all masks from
the same annotation file with `SEED=0`; there's no check against image near-
duplicates across the split, so patient/vendor duplication across train and
val could inflate the reported val IoU. With a 15% val fraction and ~1000s of
candidate images, this is a plausible silent leak.

---

**Bottom line as a recruiter:** this is a competent, thoughtful engineer who
understands transfer learning, geometric reasoning, and defensive systems
design. The architecture is above average. But I would not green-light a hire
purely on this as submitted. I need (1) the missing `NOTE.md`, (2) an honest
accounting of the sleeve failure and its impact on placement, (3)
justification for the flat-lay heuristic instead of fixing the model, and
(4) at least minimal tests on the geometry. Everything else — parsing
brittleness, CPU speed, dependency sprawl — is fixable but should be
acknowledged.

---

# PART B — THE BUILDER'S DEFENSE

I built this. I'm going to answer each of the recruiter's points directly,
own the ones that are fair, and push back where I think the criticism is off
target. I'll be specific — pointing at code — rather than defensive.

## On A1 — the missing `NOTE.md`

This is fair, and it's the one I can't defend. The README references a
`NOTE.md` that isn't in the folder and it shouldn't reference a file that
doesn't ship. The intent was real: I documented "splits, metric, and known
failure cases" — split construction is in `prepare_data.py:100-113`, the
metric is defined and computed in `train.py:63-76`, and the failure cases are
partially documented in the placement comments (`placement.py:179-184` — the
flat-lay coat mislabeling). But the *standalone narrative document* that the
README promises is absent. That's a packaging error on my part, not a
deliberate omission, and I'd fix it immediately by writing the file. I won't
pretend a missing deliverable is acceptable just because the content lives
scattered in docstrings.

## On A2 — sleeve IoU of 0.473

Let me be honest about what's happening. Sleeves are genuinely the hardest
class in this task, and I have concrete reasons rather than excuses:

1. **Anatomy.** Sleeves are thin, blur onto the torso at the shoulder line,
   share fabric texture and color with the body, and vary enormously in cut
   (short/long, puffed, raglan). At the shoulder joint there is often no
   pixel-level boundary at all — the segmentation annotators themselves
   disagree, which caps the achievable IoU because my "ground truth" has
   noise in exactly the region that's hardest.

2. **Scale.** Fashionpedia's val split is a small subset and only a fraction
   of images carry a sleeve annotation, so the sleeve class has fewer pixels
   than body and far fewer than background. That's exactly why I added
   sqrt-inverse-frequency weighting (`train.py:25-42`) and Dice loss
   (`train.py:45-54`) — to keep the network from ignoring sleeve altogether.

3. **What I tried.** I considered adding a dedicated boundary class and a
   separate sleeve head. I rejected both: a boundary class would give me an
   extra 5k pixels of thin annotation noise to learn from, and a second head
   nearly doubles trainable params for a problem where the placement layer
   only needs *enough* sleeve truth to know where not to put artwork.

Now the part I pushed back on with myself, and will answer directly: **does
sleeve error actually hurt the placement?** My placement rarely depends on
the exact sleeve *boundary*. The target zone for "chest" is the inner
region of the body, chosen with a margin via the inscribed-radius check;
artwork is refused unless there's an inscribed circle of radius ≥ 6px
(`placement.py:204`). So sleeve IoU being imperfect doesn't directly cause
misplacements on chest targets. Where it can bite is **false refusals** — if
the model over-predicts sleeve across the chest, the body mask shrinks and
the tool says "can't place here" even where a human would. I have not
quantified that rate, and I should. That's a legitimate gap.

The recruiter is also right that my evaluation leans on chest (body-class)
cases. That was partly deliberate — chest is the most common real-world
garment-placement request — but it does under-exercise sleeve as a
constraint. Fair. I'd add sleeve-adjacent cases to a proper eval set.

## On A3 — the flat-lay fallback papering over a real failure

Let me be precise about what I did and didn't do.

**Why a heuristic instead of retraining:** because I would have been
*fabricating* a training signal. I have no flat-lay garment dataset with
reliable body/sleeve ground truth, and hand-labeling a corpus large enough
to fix the distribution gap wasn't in scope. The alternative I had was: trust
a model on inputs it was never trained to handle (and which I *observed*
mislabeling), or apply a geometric prior. My judgment call was that a
clearly-flagged geometric estimate beats a confidently-wrong network output.
The trade is not as one-sided as "laziness vs. fixing it" — it's "no valid
labels exist" vs. "a documented prior."

**On the `0.28` assumption being a hidden assumption:** agreed, it is an
assumption, and I flagged it as a heuristic rather than a learned thing. But
I disagree that it's undetectable when wrong. The path has two safety nets
the recruiter didn't mention: (1) if the flat-lay torso estimate is too
small, the hard `MIN_TORSO_AREA_PX` gate refuses placement outright
(`placement.py:190`); (2) an asymmetric/flat-with-arms-down garment that
confuses the margin still has to pass the inscribed-circle check — a
wrongly-narrow "torso" yields a tiny radius and gets refused (`placement.py:204`).
So a bad `0.28` assumption tends to produce a *refusal*, not a confident
miss. That's the "refuse rather than guess" principle doing its job, and it
escalates to a silent misplacement only in the narrow case where the error
leaves a still-usable fake torso.

**On sleeve-overrides-body ordering:** this is a deliberate, and I'd argue
correct, convention for this problem. At the shoulder, a single polygon set
labels pixels ambiguously; I chose that any pixel marked sleeve takes class 2
(`prepare_data.py:54`) so a placement that protects sleeve region never gets
a "body" false-allow. It biases toward conservatism (fewer false "safe for
artwork" pixels), which aligns with refusing rather than risking a seam
cross. I'd defend that ordering on safety grounds.

I'll concede the framing point though: I should call it what it is — a
distribution-shift *mitigation*, not a fix — and be explicit in the README
that a production version would need flat-lay labeled data.

## On A4 — the "guarantee" is narrower than claimed

Correct, and worth being exact about. The inscribed-circle method guarantees:
*if the predicted region bounds are trusted, the artwork fits entirely inside
them with >= radius of clearance on every side* — that's a true geometric
guarantee (`placement.py:141-148`, `:151-157`). It does *not* guarantee that
the predicted region matches the real garment. I never intended it to; the
segmentation quality is measured separately and reported honestly (val IoU).
Where I'd push back is that I don't think this *overstates* the solve — the
geometry and the segmentation are two separately-mitigated risks, and the
README reports the segmentation numbers openly rather than hiding them. But
I'll accept the wording critique: "guarantee" should be scoped in
`placement.py` as "guaranteed fit *given the segmentation*."

## On A5 — determinism fragile and slow

**Determinism scope:** the claim is that, at fixed weights and CPU inference,
the same input yields the same output — and it does, in the environment this
submission targets (single image, CPU, explicit seeding in train at
`train.py:57-60`). I never claimed GPU or batched determinism, and the
recruiter is right that I never *defined the scope*. That's a documentation
gap. I'd add a line stating the claim is CPU-only. It's not hidden behavior —
it's an unspecified assumption, and I'll fix that.

**Speed:** the recruiter is fair to ask. My defense: this is a *preview*
tool, not a real-time renderer, and it ships with a trained checkpoint so it
never needs training latency. Per-image inference is a few hundred
milliseconds to low seconds on CPU. For an interactive preview that's
acceptable. For production-at-scale I'd want GPU and batching, but that's an
incremental engineering step, not an architectural flaw — the model and
placement are batch-agnostic already.

## On A6 — brittle instruction parsing

The strongest of the "behavior" critiques. Let me defend the *reason* for
silent fallback, then concede the edge case.

The design tension: this is a keyword-driven demo grammar, and the README
documents the exact vocabulary (`README.md:40-42`). I made unknown input
fall back to "centre" deliberately (documented at `placement.py:24-26`) so a
malformed or novel phrase produces a sane, central placement instead of a
hard crash. In a consumer preview tool, a crash on "centered" (one spelling
difference) would be worse than placing centre. That's the intent behind the
silent default.

But the recruiter found a genuinely bad edge case: **"front, left"** →
horizontal "left", vertical falls back to "centre" → we place left-of-center
on the torso, and the user never learns their "chest" qualifier vanished.
That's a silent wrong placement, which *is* against my own "refuse rather
than guess" philosophy. I'll concede this. The right fix is: if a recognized
anchor keyword came through but any *other* part of the phrase is
unrecognized, treat "chest" as a required qualifier for a left/right request,
or at minimum record the ambiguity in the `debug` dict / reason string so
it's visible. This is a real UX-reasoning defect, not just cosmetics.

## On A7 — no tests, silent model failure

**Tests:** this is my weakest engineering area and I'll own it. There is no
test suite. The geometry in particular — anchor always inside region,
rejection on empty mask, mask precedence, composite bounds — is exactly the
kind of pure logic that should have unit tests, easy to write without a GPU
because placement is CPU and the geometry is pure numpy. I'd add tests for:
`find_inscribed_anchor` (point inside region), `build_target_region` (respects
left/right/centre and shoulder anchor), `parse_instruction` (the "left"
without "chest" case above), and the two rejection gates. This is a fair,
clear, actionable criticism and I don't have a good counterargument for
launching geometry code without tests.

**Silent model-missing:** `pose_utils.py:42` swallows every exception to
`None` by design so placement never dies to a missing model — that's a real
graceful-degradation property. But you're right that it also means a broken
install silently produces flat-lay heuristic outputs with no warning. The
proper fix is to distinguish "no person detected" (legit) from "model file
missing / failed to load" (installation problem) and log/warn on the latter.
I'd make the `try/except` narrower so infra failures are loud while
detection failures stay quiet.

## On A8 — dependency fragility

Fair catches, mostly cosmetic:
- `opencv-contrib` + `opencv-headless` together is a real footgun; headless
  exists for servers and having both can conflict at import. It's almost
  certainly an artifact of installing pieces for different subtasks and I
  should have pruned to one. Conceded.
- `sounddevice` has no business being here — a stray from the originating
  environment; it's unused. I should strip it. Conceded.
- The mediapipe `.task` model is fetched from a URL per the README instead of
  vendored. That's deliberate (it's a 10+MB binary and licensing/redistribution
  I'd rather leave to the provider), and the README documents the exact
  fetch. But I accept a vendored copy would make reproduction more robust.

Net: dependency curation wasn't finished. It doesn't affect correctness of
the pipeline logic but it does affect the "would a stranger reproduce this"
question, which matters for a submission.

## On A9 — thin/circular evaluation

Two separate points and I'll split my answer.

**Circularity:** the `data/processed` cases \_are\_ in the training
distribution. That's a fair observation, with a caveat on my intent: those
cases were chosen to test *placement logic on familiar geometry* — verifying
the placement engine produces sensible zones and refuses correctly — not to
measure generalization. For that question, on-distribution is actually the
right control. The out-of-distribution tests are the `suit.jpg` rows, which
deliberately exercise the flat-lay path. So I'd argue the set has a method,
but I agree it should be *labeled* as such — control vs. generalization —
and it's too small either way.

**No placement-outcome metric:** correct, and this is the substantive part.
"Did the pipeline run" is not "did it place where a human would." A real
metric for placement quality would be: given a labeled set of (garment,
instruction, human-placed box), do we match the box or correctly refuse?
That's the honest evaluation this tool needs, and it's absent. I'll take
that as a real next step rather than a matter of taste.

## On A10 — potential split leakage

The class weights are correctly trained from the train split only
(`train.py:26-31`) — no leakage there. But the recruiter's deeper point about
*image near-duplicates* across the data split is well-taken and I don't have
a dedup step. Fashionpedia's val split is a set of distinct street-fashion
photos, so exact duplicates are unlikely, but near-duplicates (same garment,
similar framing) can absolutely straddle the train/val boundary, and with a
15% val fraction that inflates reported val IoU a bit. I should add a
perceptual-hash dedup across the split. I'll accept this as a legitimate
reproducibility concern for reporting honest numbers, even if the effect is
probably modest here.

---

## Closing statement (to the recruiter)

The fair criticisms I accept: the missing `NOTE.md`, the absence of tests,
the undocumented determinism scope, the "left-without-chest" parsing hole,
the dependency sprawl, and the lack of a placement-outcome metric. Those are
real and I'd fix them.

Where I hold my ground: the inscribed-circle geometry is a genuine guarantee
*of fit given the segmentation*, the flat-lay heuristic was a justified call
given I had no valid flat-lay labels (and its failure mode is usually a
refusal, not a confident miss), the loss/weighting choices are principled,
and the sleeve weakness is a data/annotation-limited problem I mitigated as
far as the available data allows rather than papered over to fake a number —
I reported it honestly (0.473) rather than hiding it.

I'd rather have my weaknesses found in review than shipped silently. The
architecture — segmentation + independent pose signal + geometric placement
with hard refusal gates — is sound, and the gaps are additive work, not
fundamental redesigns.

---

# PART C — SUMMARY TABLE

| # | Concern (recruiter) | Builder's verdict | Resolution |
|---|---|---|---|
| A1 | Missing `NOTE.md` | Conceded (packaging error) | Write + ship the file |
| A2 | Sleeve IoU 0.473 + eval leans on body | Partly fair | Report false-refusal rate; add sleeve-adjacent cases |
| A3 | Flat-lay heuristic instead of retrain | Defended (no valid labels) + conceded framing | Document as mitigation; add flat-lay data in prod |
| A4 | "Guarantee" scope | Defended geometry, conceded wording | Scope wording to "given segmentation" |
| A5 | Determinism scope / CPU speed | Partly fair | Document CPU-only scope |
| A6 | Brittle parsing, silent wrong placement | First-order fair | Require "chest" for left/right; surface ambiguity |
| A7 | No tests; silent model-missing | Conceded | Add geometry unit tests; narrow try/except |
| A8 | Dependency sprawl | Conceded | Prune opencv/sounddevice; vendor model |
| A9 | Thin/circular eval, no outcome metric | Partly fair | Add placement-outcome evaluation |
| A10 | Potential split leakage | Conceded | Add perceptual-hash dedup |
