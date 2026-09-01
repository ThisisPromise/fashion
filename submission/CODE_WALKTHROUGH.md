# Code Walkthrough — Garment segmentation and artwork placement

A recruiter-facing deep dive into the submission, written to help the
candidate defend the code in an interview and to help the recruiter probe
real understanding.

---

## The problem it solves (start here — it's the "why")

The tool places artwork (a logo) on a garment photo. The **naive approach**
is hardcoded pixel offsets — "put the logo at (x=200, y=150)". That breaks
instantly: a different garment size, a different pose, or a different crop
shifts where "left chest" actually is. The logo lands on a sleeve or crosses
a seam.

The key requirement this submission is built around: **"left chest" must be
defined relative to the garment's actual geometry in the photo**, not
relative to fixed pixels. And critically — the README states — **if it can't
find a good spot, it must refuse rather than guess**. That "refuse rather
than guess" is the single most important design principle in the whole
codebase, and it shows up repeatedly.

This is done in two parts:
1. **A segmentation model** (`model.py`, `train.py`, `prepare_data.py`,
   `dataset.py`) that classifies each pixel as background / body / sleeve.
2. **A placement engine** (`placement.py`) that uses those regions plus a
   pose detector to compute where a region *actually is*, and refuses if
   it's not usable.

---

## Part 1 — The segmentation pipeline

### `prepare_data.py` — building the dataset from Fashionpedia

The whole thing is supervised segmentation, so it needs masks. The clever
part is that there's **no new labeling done** — it derives 3-class masks
from Fashionpedia's existing polygon annotations:

- It keeps only images that have **both** an upper-body/whole-body garment
  **and** a sleeve annotation (`qualifying_ids`, `prepare_data.py:71-74`).
  This guarantees every training image exercises the hardest class (sleeve).
- `build_target_mask` rasterizes polygons into a 3-class mask. Note the
  precedence at `prepare_data.py:54`: `target[sleeve] = 2` — **sleeve
  overrides body in overlap**. This is a deliberate design decision about
  ambiguity at the shoulder.

One thing to have them defend: **they don't train a "boundary" or "seam"
class at all** — the file docstring explicitly says boundary is derived at
inference time rather than learned. That's a simplification chosen because
the placement logic only needs body vs. sleeve, not pixel-perfect seams. Ask
them why they chose 3 classes instead of 4+ (seams, collars, hems) and
whether that hurt sleeve IoU.

There's also a **sanity check** at `prepare_data.py:116-123` that prints
class pixel coverage over 50 images — an empty-mask bug would surface
immediately. This is a good sign of engineering discipline.

### `dataset.py` — loading + augmentation

Standard `torch.utils.data.Dataset`. The augmentations are **geometric +
photometric, applied in mask-space together** so the mask stays aligned with
the image:
- Random scale-crop (0.85–1.0)
- Random horizontal flip (critical for segmentation — flip both image and
  mask together)
- Brightness/contrast jitter

The important detail: **NEAREST interpolation for masks when resizing**
(`dataset.py:48,68`). If you used bilinear on a mask, you'd smear class
boundaries and blend adjacent class IDs into garbage values. This is a
classic implementation trap they got right.

### `model.py` — TinyGarmentSegModel (the gem to grill them on)

This is the strongest architectural piece.

**The constraint: 2M trainable parameters.** The README and `train.py:121`
enforce this with an assert. To meet it, they use a **transfer-learning
trick**:
- A **pretrained MobileNetV3-Small backbone**, which is **frozen** (all
  `requires_grad = False`, `model.py:73-74`).
- Only a small trainable **FPN-style decoder** is trained.
- Result: **152,323 trainable params** vs **927,008 frozen** (from
  val_metrics.json), comfortably under the 2M cap.

The reason this works is the key ML insight to defend: **a frozen
ImageNet-pretrained backbone already produces features that separate
garments from background** — low-level edges, textures, color. What needs to
be *learned* is only the task-specific mapping (3-class segmentation), which
sits in the decoder. So you freeze the expensive part and only train a cheap
head. This is textbook transfer learning.

**The FPN decoder (`LightDecoder`)** — multi-scale fusion:

```
Backbone -> taps at strides 4, 8, 16 (FEATURE_TAPS = {low:3, mid:6, high:12})
```

Semantic segmentation needs both **fine detail** (sleeve edges — low stride)
and **semantic context** (is this whole region the body? — high stride). A
single-scale head can't do both. The FPN fuses three feature maps
**top-down** (`model.py:56-60`): it upsamples the deepest/most-semantic map,
adds it to the mid map, smooths, upsamples again, adds to the low map. This
gives each pixel info from all scales.

**Two subtle engineering details worth probing:**

1. **Channel counts read empirically, not hardcoded** (`model.py:86-89`):
   they run a dummy tensor through the backbone and read `f.shape[1]` to get
   the tap channels. This means the code stays correct if torchvision
   changes MobileNetV3's internals. This shows they think about code
   robustness, not just "make it work today."

2. **Frozen BN layers stay in eval mode** (`model.py:106-112`): this is a
   *subtle and correct* handling of BatchNorm with frozen weights. BatchNorm
   has running statistics that update during training. If you freeze the
   backbone layers but leave BN in train mode, the running mean/var keep
   updating — but their scale/gamma are frozen, which can be inconsistent.
   By forcing frozen layers into eval mode
   (`layer.train(mode and i in self.unfrozen_layer_indices)`), they keep the
   ImageNet statistics fixed. **This is a genuinely non-obvious
   implementation detail** — many people get frozen-BN wrong. Definitely
   probe this.

3. The logits are upsampled back to *input* size at the end
   (`model.py:118`), so the head predicts at stride 4 but the loss is
   computed at full resolution.

### `train.py` — training

Several defensible choices:

- **Optimizer split by parameter group** (`train.py:129-132`): unfrozen
  backbone layers get `lr * 0.1`, decoder gets full `lr`. Standard
  fine-tuning practice — pretrained weights are close to good, so a
  full-strength LR could destroy them; you nudge them gently.
- **Loss = CrossEntropy + Dice** (`train.py:138-141`). This is a principled
  combination: CE gives a stable, well-calibrated gradient signal; **Dice
  directly optimizes the metric you care about** (overlap / union) and
  handles class imbalance better. On a dataset where background dominates
  (that's why background IoU is 0.959), a pure CE loss can underweight rare
  classes like sleeve. Dice softens that.
- **Sqrt-inverse-frequency class weighting** (`train.py:25-42`): computed
  from the *train split only* (no data leakage into val). The `1/sqrt(freq)`
  damping prevents the minority class from being overshone, while
  `weights / weights.mean()` keeps the loss magnitude comparable to
  unweighted CE so the learning rate still behaves.
- **Confusion-matrix IoU evaluation** (`train.py:63-76`): IoU computed from
  a vectorized `torch.bincount` — clean and fast, handles NaN for absent
  classes.
- **Early stopping = best val loss checkpoint** (`train.py:174-183`), and
  final metrics are recomputed from the *best* checkpoint, not the last
  epoch (`train.py:186-188`). Correct — this is how to report honest numbers.

---

## Part 2 — The placement engine (`placement.py`, the other big one)

This is where "refuse rather than guess" lives. The pipeline is:

```
predict_regions -> get_shoulder_y -> choose torso mask -> parse instruction
  -> build_target_region -> find_inscribed_anchor -> fit_artwork_to_radius -> composite
```

### `predict_regions` (`placement.py:51-69`)

Runs the model at training resolution (320), then **upsamples the argmax
prediction (NEAREST) back to the original image resolution**. Nearest is the
right choice — it preserves the hard class label exactly; bilinear would
blend class IDs into fractional nonsense. Then placement geometry is
computed in the caller's *actual pixel space*, so it's not dependent on a
resize.

### The flat-lay fallback (`placement.py:179-184`) — a really thoughtful part

Here's the subtle real-world problem they hit and solved. The model was
**trained almost entirely on worn garments** (Fashionpedia images are people
wearing clothes). But the demo includes `sample_garments/suit.jpg` — a
**flat lay or hanger photo with no person**.

The comment at `placement.py:182` documents a real observed failure: *a real
flat-lay coat photo got labeled mostly "sleeve"* because the model never saw
that distribution. For a front-facing flat lay, the whole garment is
basically "body" from the model's perspective except the outer edges.

So they detect this case via the **pose model**:
- `get_shoulder_y` uses MediaPipe PoseLandmarker to find shoulder landmarks
  (`pose_utils.py`).
- **If a person is detected** (shoulders visible, both confident,
  `placement.py:176`), trust the model's body/sleeve split.
- **If no person is detected** (`shoulder_y is None`), fall back to a
  **geometric estimate** (`infer_flat_lay_torso`): treat the outer ~28% of
  the silhouette's bounding box as "sleeve" and the rest as "torso". A
  reasonable heuristic — a flat-lay garment's sleeves splay out to the
  sides.

This is genuinely good thinking: **use a second, independent signal (pose)
to know when your primary model is out of distribution, and degrade
gracefully instead of trusting a nonsense prediction.** Worth probing in an
interview.

### `build_target_region` (`placement.py:105-138`) — instruction meets geometry

This converts text like "front, left chest" into a pixel mask on the torso:

- `parse_instruction` does **plain keyword matching**
  (`placement.py:72-83`): any phrase containing
  "left"/"right"/"centre", "chest"/"breast", "front"/"back". The docstring
  admits it's *not* language understanding, and unrecognized phrases
  **silently fall back to "centre"** — a documented limitation rather than a
  hidden bug. Ask why silent fallback over erroring (answer: robustness —
  a slightly-off placement beats a hard crash in production UX).
- **Shoulder-line anchoring** (`placement.py:116-121`): the *clever* bit.
  The naive way to find the top of the chest is to use the torso mask's
  bounding-box top. But **a raised arm pulls the bbox top upward toward the
  hand**, not the shoulder. So they use the pose-detected shoulder line as
  the true top of the chest instead. This is precisely the "don't use fixed
  pixels, use real geometry" philosophy. If no shoulder is available, they
  fall back to a `0.22` margin estimate.
- The chest zone is defined as the top 45% of the torso below the shoulder;
  horizontal zones are side fractions (28–40%) or a middle band. All are
  **relative fractions of the detected region**, never absolute pixels.

### `find_inscribed_anchor` (`placement.py:141-148`) — the mathematical core

The part most worth seeing defended, because it's the least "standard ML"
piece:

```python
dist = ndi.distance_transform_edt(region_mask)
cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
```

A **Euclidean distance transform** computes, for every pixel in the target
region, the distance to the **nearest boundary pixel**. Its *maximum* is the
**center of the largest circle that can be inscribed inside the region** —
the biggest circle that fits without crossing the region boundary. Placing
the artwork at that center, scaled so its half-diagonal fits within that
radius (via `MAX_ARTWORK_FRACTION_OF_RADIUS`, `placement.py:151-157`),
**guarantees the artwork stays inside the region — it cannot cross a seam or
spill onto a sleeve.**

This is a beautiful, deterministic geometric guarantee: instead of heuristic
"nudge it a bit," the anchor is *the* pixel guaranteed to have the most
clearance, and the artwork is *provably* scaled to fit. And it's
**deterministic** — same input, same model, same output (determinism
explicitly stated at `placement.py:168-170`).

### Refusing to guess — the safety checks

Two hard-failure gates enforce "refuse rather than guess":

1. **Torso too small/absent** (`placement.py:190-197`):
   `torso_mask.sum() < MIN_TORSO_AREA_PX` (400 px) → reject with a
   human-readable reason.
2. **Target zone too small/thin** (`placement.py:204-213`):
   `radius < MIN_INSCRIBED_RADIUS_PX` (6 px) → reject, because placing
   artwork there risks seam crossing. The error message even says whether
   the flat-lay fallback was already tried.

Notably, `MIN_TORSO_AREA_PX` and `MIN_INSCRIBED_RADIUS_PX` are **named
module-level constants** with explanatory comments — clear, tunable,
documented. The mark of code written for maintenance, not a one-off.

### The result object (`PlacementResult` dataclass)

Every path returns a `PlacementResult` with `status` ("ok" or "rejected"), a
human `reason`, and optional debug payloads (raw masks, target region,
anchor point, fitted size, and a `debug` dict). A clean, uniform contract
that lets the demo, evaluation script, and future callers handle both
success and rejection gracefully without try/except gymnastics. It also
means **rejection is a first-class outcome**, not an error — exactly the
"refuse rather than guess" design.

---

## Part 3 — Supporting scripts

- **`pose_utils.py`**: thin wrapper around MediaPipe PoseLandmarker. Key
  design point (`pose_utils.py:42`): the whole function is wrapped in
  `try/except` and returns `None` on *any* failure (no person, low
  confidence, model missing, crash). The pose detector **degrades to "not
  available" rather than crashing placement** — treated as optional, falls
  back. Defense in depth.
- **`run_evaluation.py`**: runs every row of `evaluation_cases.csv`, renders
  before/after, and writes a `_summary.csv` with status/reason/anchor for
  each case. The rejection card (`rejection_placard`) makes failures
  *visible* rather than silently skipped.
- **`demo.py`**: the interactive single-image path.
- **`viz.py`**: shared overlay color-coded renders.
- **`visualize_predictions.py`**: GT vs. prediction side-by-side, so you can
  *see* where the model fails, not just read a number. Numbers + eyeballs is
  good evaluation practice.

---

## What to grill them on in the interview (priority order)

1. **The frozen-BN eval-mode handling** (`model.py:106-112`) — explain *why*
   frozen BatchNorm must not be left in train mode. If they can articulate
   running-statistics vs. trainable-parameters, they know their torch well.
2. **The inscribed-circle anchor** (`placement.py:141-148`) — walk through
   the distance transform and the geometric guarantee it gives. This
   separates someone who understands the math from someone who copied it.
3. **The flat-lay vs. worn fallback** (`placement.py:179-184`) — why trust
   the pose model to decide which segmentation to believe? This shows
   systems-level thinking (knowing your model's distributional limits).
4. **Dice + CE loss and sqrt-inverse-frequency weighting** — why both, and
   why the damping?
5. **The 2M parameter strategy** — why freeze most of a pretrained backbone
   and train only a decoder? What does the frozen backbone already "know"?

---

## Known weak spots / where to push back

- **Sleeve IoU is low** (0.473). The model genuinely struggles with sleeves —
  the README admits this. Ask why, and whether the flat-lay fallback masks
  that deficiency in the demos.
- **Keyword parsing is brittle** — "left chest" without a comma, or "left
  sleeve", would be misparsed. It's documented, but it's the most fragile
  part of the UX.
- **`NOTE.md` referenced in the README doesn't exist.** The README says
  failure cases are documented "in NOTE.md" (README.md:21 and :101), but
  there's no NOTE.md file in the folder. A **discrepancy worth flagging** —
  either omitted from the submission or the doc is stale.
- **Determinism depends on the CPU path** — the whole training/prediction
  runs on CPU (device hardcoded to "cpu"), which is fine for this size but
  worth asking about for scaling.
