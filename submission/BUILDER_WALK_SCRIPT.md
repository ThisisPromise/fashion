# BUILDER'S CODE-WALK SCRIPT

A self-paced script for the builder to explain the project end-to-end, with
the *why* behind every decision, pointing to real code lines. Use this to
rehearse before the interview or as a live walkthrough guide.

How to use: each section gives you (1) what to say, (2) the exact code to
open/point to, and (3) the "why" — the reasoning that answers the question
before it's asked.

---

## 0. The 30-second pitch (say this first)

> "This tool places artwork on garment photos. The old way used fixed pixel
> offsets — 'put it at x=200, y=150' — which broke whenever the garment
> changed size, pose, or crop, and the logo would drift onto sleeves or
> cross seams. So I built two things: a segmentation model that finds the
> body panel and sleeves, and a placement engine that works out where 'left
> chest' actually is on *that specific photo* using geometry — and refuses
> to place if it can't find a safe spot. No hardcoded offsets."

---

## 1. The problem & the guiding principle

**Say:** "The core requirement is that 'left chest' is defined relative to
the garment in the photo, not to fixed pixels. And critically — if the
system can't find a good spot, it must **refuse rather than guess**. That
principle runs through the whole design."

**Point to:**
- `README.md:13-18` — the problem statement and the two-part solution.
- `README.md:64-65` — the "refuse rather than guess" idea, stated as a feature.
- `src/placement.py:16-18` — the constants `MIN_INSCRIBED_RADIUS_PX`,
  `MAX_ARTWORK_FRACTION_OF_RADIUS`, `MIN_TORSO_AREA_PX`; these are the named
  thresholds where the system says "no."

**Why they did it:** A placement that lands on a seam/sleeve is worse than
no placement. Refusing is a deliberate UX/product decision, not an accident.

---

## 2. Data: `src/prepare_data.py` — building masks without hand-labeling

**Say:** "I needed pixel masks for body and sleeve, but I didn't label
anything by hand. I derived 3-class masks from Fashionpedia's existing
polygon annotations."

**Point to:**
- `src/prepare_data.py:21-22` — `UPPERBODY_AND_WHOLEBODY_GARMENT_IDS` and
  `SLEEVE_ID = 31`. I only keep the garment categories relevant to this task.
- `src/prepare_data.py:71-74` — `qualifying_ids`: keep an image only if it
  has *both* an upper/whole-body garment AND a sleeve annotation. This
  guarantees every training image exercises the hardest class (sleeve).
- `src/prepare_data.py:37-55` — `build_target_mask`: rasterize polygons;
  class 1 = body, class 2 = sleeve.
- `src/prepare_data.py:54` — `target[sleeve] = 2` **sleeve overrides body in
  overlap.** Explain below.
- `src/prepare_data.py:100-113` — the train/val split (`SEED=0`, 15% val).

**Why decisions:**

- **3 classes only (background/body/sleeve), no seam class.** The placement
  layer only needs body-vs-sleeve to know where artwork can go. A dedicated
  seam/boundary class would add annotation noise and params without helping
  placement. This is a deliberate simplification.

- **Sleeve overrides body (`:54`):** at the shoulder, pixels are ambiguous.
  I chose that any pixel marked sleeve becomes "sleeve" so a placement
  protecting the sleeve region never gets a false "this is safe body." It
  biases toward conservatism — fewer false "safe for artwork" pixels — which
  aligns with refusing rather than risking a seam cross.

- **Splitting with a fixed seed:** guarantees the same split every run, so
  training is reproducible.

**Show the verification (important — recruiter asked about leakage):**
I ran a check: the split has **0 overlapping image IDs** between train and
val (630 train / 111 val). And the `evaluation_cases.csv` images are all in
*validation* or external — **none were used in training**, so the eval tests
never-before-seen images, as intended.

---

## 3. Data loading & augmentation: `src/dataset.py`

**Say:** "Standard `torch.utils.data.Dataset`, but the critical detail is
that augmentation happens on the image *and* mask together, and masks use
nearest-neighbour interpolation."

**Point to:**
- `src/dataset.py:30-40` — dataset structure.
- `src/dataset.py:44-54` — `__getitem__`: loads image + mask, normalizes with
  ImageNet stats (because the backbone was trained on ImageNet).
- `src/dataset.py:56-77` — `_augment`: random scale-crop, horizontal flip,
  brightness/contrast jitter.

**Why decisions:**

- **Nearest interpolation for masks** (`:48`, `:68`): bilinear would blend
  neighbouring class IDs into garbage fractional values and smear boundaries.
  Nearest keeps hard class labels intact. Classic trap, easy to get right.
- **Flip both image and mask together** (`:70-72`): a segmentation mask is
  only valid if it stays aligned with the image. Never flip one without the
  other.
- **ImageNet normalization** (`:51`): the backbone was pretrained on
  ImageNet with these exact mean/std, so the input must be normalized the
  same way for the frozen features to be meaningful.

---

## 4. The model: `src/model.py` — TinyGarmentSegModel (the gem)

**Say:** "The task has a hard cap of ~2M trainable parameters. I meet it with
transfer learning: freeze a pretrained MobileNetV3-Small backbone and only
train a small FPN-style decoder."

**Point to:**
- `src/model.py:65-92` — `TinyGarmentSegModel.__init__`.
- `src/model.py:73-74` — freeze all backbone params (`requires_grad = False`).
- `src/model.py:76-82` — optional unfreezing of the last N layers.
- `src/model.py:34-62` — `LightDecoder`, the trainable FPN head.
- `src/model.py:20` — `FEATURE_TAPS` taps at strides 4/8/16.
- `src/model.py:106-112` — the frozen-BN handling (see below).

**Why decisions:**

- **Freeze a pretrained backbone:** a frozen ImageNet backbone already
  produces features that separate garments from background (edges, texture,
  colour). Only the task-specific mapping (3-class segmentation) needs to be
  learned, and that lives in the decoder. Result: **~152k trainable params**
  vs **~927k frozen** — comfortably under the 2M cap (`outputs/val_metrics.json`).
- **FPN-style multi-scale decoder:** segmentation needs both fine detail
  (sleeve edges = low stride) and semantic context (is this whole region the
  body? = high stride). The decoder fuses three feature maps top-down
  (`:56-60`): upsample the deepest map, add to the mid, smooth, upsample
  again, add to the low. Every pixel gets info from all scales.
- **Channel counts read empirically** (`:86-89`): I run a dummy tensor
  through the backbone and read `f.shape[1]`. This keeps the code correct if
  torchvision changes MobileNetV3 internals — robust, not hardcoded.
- **Frozen BatchNorm stays in eval mode** (`:106-112`): *this is the subtle
  one.* BatchNorm has running statistics that update during training. If I
  freeze the backbone but leave BN in train mode, the running mean/var keep
  drifting while the scale/gamma stay frozen — inconsistent. By
  `layer.train(mode and i in self.unfrozen_layer_indices)` I keep the frozen
  layers' ImageNet statistics fixed. Many people get frozen-BN wrong; this is
  correct.
- **Logits upsampled to input size** (`:118`): head predicts at stride 4, but
  the loss is computed at full resolution so boundaries get full-grad.

**Show the headline number honestly:**
> "Mean IoU 0.693 (background 0.959, body 0.647, sleeve 0.473) on
> validation. Background is easy; body is decent; sleeves are the weak
> point — I'll be upfront about that."

---

## 5. Training: `src/train.py`

**Say:** "Four principled choices: a split optimizer, a combined loss, class
weights, and honest checkpoint selection."

**Point to:**
- `src/train.py:126-132` — two optimizer param groups: decoder = full LR,
  unfrozen backbone = LR * 0.1 (`backbone_lr_multiplier`).
- `src/train.py:134-141` — the loss: CrossEntropy + Dice.
- `src/train.py:25-42` — `compute_class_weights`: sqrt-inverse-frequency,
  from the **train split only**.
- `src/train.py:45-54` — `dice_loss`.
- `src/train.py:63-76` — confusion-matrix IoU.
- `src/train.py:174-188` — save best-val-loss checkpoint, re-evaluate it.

**Why decisions:**

- **Separate LR for backbone vs decoder:** unfrozen pretrained weights are
  close to good already; a full-strength LR could distort them. I nudge them
  gently (0.1x) while the decoder learns from scratch at full LR.
- **CE + Dice:** CE gives stable, well-calibrated gradients; Dice directly
  optimizes the overlap metric and handles class imbalance (background
  dominates). Together they're more robust than either alone.
- **sqrt-inverse-frequency weights:** background is most pixels, so pure CE
  would underweight rare classes like sleeve. `1/sqrt(freq)` keeps the
  minority class relevant, and normalising by the mean keeps loss magnitude
  comparable to unweighted CE so the LR still behaves. Computed from train
  **only** — no validation leakage.
- **Confusion-matrix IoU:** vectorised `torch.bincount` — clean, fast, and
  naturally handles absent classes (NaN).
- **Report the BEST checkpoint, not the last epoch** (`:186-188`): honest
  numbers; last-epoch metrics can be worse than the best model.

---

## 6. Placement: `src/placement.py` (the second big one)

**Say:** "Segmentation is only half the job. The other half is turning 'front,
left chest' into a *location*, and refusing when there isn't a safe one."

### 6a. Inference & upsampling — `predict_regions` (`:51-69`)
- Model runs at training resolution (320); the argmax prediction is upsample
  back to the **original image size with NEAREST** (`:67`). Nearest preserves
  hard class labels exactly (bilinear would blend classes into nonsense).
  Placement then runs in the caller's real pixel space — resolution-agnostic.

### 6b. Pose fallback — `:174-184` and `src/pose_utils.py`
**Say:** "The model was trained almost entirely on worn garments. For a
*hanger/flat-lay* photo there's no person, and the model mislabels those —
I observed a coat getting marked mostly 'sleeve'. So I bring in an
independent signal: MediaPipe pose."

**Point to:**
- `src/placement.py:174-177` — run pose; `shoulder_y` comes from
  `get_shoulder_y`.
- `src/placement.py:179-184` — if no person (`shoulder_y is None`), fall back
  to a geometric torso estimate from the raw silhouette.
- `src/placement.py:86-102` — `infer_flat_lay_torso`: treat the outer ~28% of
  the silhouette bbox as "sleeve", the rest as "torso".
- `src/pose_utils.py:27-42` — `get_shoulder_y`, wrapped in try/except.

**Why decisions:**
- **Use pose as a second, independent signal:** it tells me when my *primary*
  model is out of distribution (no person = flat-lay), so I don't trust a
  nonsense prediction. That's the "know your model's limits" mindset.
- **Flat-lay geometric heuristic instead of retraining:** I have no flat-lay
  dataset with valid body/sleeve labels, so retraining would mean fabricating
  a signal. A clearly-flagged geometric prior beats a confidently-wrong
  network output. NOTE: this is a *mitigation*, not a fix — production would
  want flat-lay labeled data.
- **Graceful degradation (`pose_utils.py:42`):** any pose failure returns
  `None` rather than crashing, so placement always has an answer path.
  (Caveat: I'd make 'model missing' loud vs 'no person' quiet — infra failure
  should warn, detection failure should not.)

### 6c. Instruction → region — `build_target_region` (`:105-138`)
**Say:** "This turns text into a pixel mask on the torso."

**Point to:**
- `src/placement.py:72-83` — `parse_instruction`: plain keyword matching
  (left/right/centre, chest/breast, front/back). Unknown → silent "centre".
- `src/placement.py:116-121` — **shoulder-line anchoring.** The naive way to
  find the top of the chest is the mask bbox top — but a *raised arm pulls
  the bbox top up toward the hand*, not the shoulder. So I anchor the chest
  zone to the pose-detected shoulder line instead. This is the "real
  geometry, not fixed pixels" philosophy in action.
- `src/placement.py:124-138` — chest = top ~45% of torso below shoulder;
  horizontal zones are side fractions or a middle band. All relative to the
  *detected region*, never absolute pixels.

### 6d. The anchor — `find_inscribed_anchor` (`:141-148`) — the math core
**Say:** "This is the part I'm proudest of. Instead of 'nudge it until it
looks right', I compute the *mathematically guaranteed* safest point."

**Point to:**
```python
dist = ndi.distance_transform_edt(region_mask)
cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
```
**Explain:** A Euclidean distance transform gives each pixel its distance to
the nearest boundary. Its maximum is the centre of the **largest circle that
fits inside the region**. I place the artwork there, scaled so its
half-diagonal ≤ that radius (`fit_artwork_to_radius`, `:151-157`) → **the
artwork provably cannot cross the region boundary or spill onto a sleeve.**

**Caveat to volunteer:** this guarantees fit *given the segmentation* — the
geometry is exact, the segmentation itself is imperfect (see sleeve IoU). So
call it "guaranteed fit if the region is right", not an absolute guarantee.

### 6e. Refusing to guess — the hard gates
- `:190-197` — `MIN_TORSO_AREA_PX` (400): if the predicted body area is too
  small/absent → reject with a human-readable reason.
- `:204-213` — `MIN_INSCRIBED_RADIUS_PX` (6px): if the requested zone is too
  thin → reject, because placing there risks a seam crossing.
- `:29-39` — `PlacementResult` dataclass returns status + reason + debug on
  **every** path, so rejection is a first-class outcome, not an error.

### 6f. Determinism — `:168-170`
**Say:** "Same input, same model, same output — no randomness anywhere on
this path." (Scope honestly: this holds for CPU single-image inference, which
is the submission's target. I'd document it as CPU-only.)

---

## 7. The surrounding tools (brief)

- `src/run_evaluation.py` — runs every `evaluation_cases.csv` row, renders
  before/after, writes a `_summary.csv` with status/reason/anchor.
- `src/demo.py` — interactive single-image path.
- `src/viz.py` — shared colour-coded overlay helpers.
- `src/visualize_predictions.py` — GT vs prediction side-by-side, so you *see*
  where the model fails, not just read a number.

---

## 8. Honest weaknesses to volunteer (say these before asked)

1. **Sleeve IoU 0.473 is the weakest class.** Why: sleeves are thin, blur onto
   the torso at the shoulder, and the annotations themselves are noisy there.
   My mitigation: sqrt-inverse-frequency weights + Dice, and placement never
   depends on the exact sleeve boundary (it uses an inscribed circle with
   margin). Open gap: I haven't quantified the *false-refusal* rate when
   sleeve is over-predicted across the chest.
2. **The flat-lay fallback is a heuristic, not a fix.** It assumes sleeves
   splay to the outer ~28% of the bbox. Its failure mode usually produces a
   *refusal* (city-gate + inscribed-radius checks), not a confident miss —
   but production needs real flat-lay data.
3. **Keyword parsing is brittle.** "front, left" silently becomes
   "left-centre" without telling the user their "chest" vanished. I'd require
   "chest" for left/right or surface the ambiguity.
4. **No test suite yet.** The geometry (anchor-inside-region, rejection on
   empty mask, precedence, composite bounds) should have unit tests. Pure
   numpy/CPU, easy to add — this is the fair criticism I most agree with.
5. **Dependency sprawl** (`requirements.txt`): `opencv-contrib` +
   `opencv-headless` together and a stray `sounddevice` — should be pruned.
6. **Eval is on-distribution.** The `data/processed` eval cases are
   Fashionpedia val photos (same distribution as training) — good for testing
   placement logic, but only `suit.jpg` is truly out-of-distribution.
7. **`NOTE.md` is referenced in the README but wasn't shipped.** Packaging
   error — I'd write and add it.

---

## 9. The one-liner for "what did you actually build?"

> "A transfer-learned segmentation model that finds the body and sleeves of a
> garment in a photo, plus a deterministic placement engine that uses pose +
> distance-transform geometry to place artwork at the safest in-region point
> — and refuses, with a clear reason, whenever there isn't a safe spot."
