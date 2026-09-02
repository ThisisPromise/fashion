# Live Presentation Script — Garment Segmentation & Placement

Spoken script, first person, written to be said out loud. `[SHOW: file]` /
`[RUN: command]` are stage directions for what to have on screen. Roughly
12-15 minutes at a natural pace; trim the training-journey section first if
you're short on time, it's the most compressible.

---

## 1. Open with the problem (30-45 sec)

**Say:**

> "The brief was about a garment preview tool that places artwork on photos.
> The existing approach used fixed pixel coordinates — 'left chest' always
> meant the same x, y — so it had no idea what a chest panel or a sleeve
> actually was. It was just guessing coordinates. If the garment was a
> different size, pose, or crop, the logo would drift onto a sleeve or cross
> a seam.
>
> I built two things to fix that: a small model that segments a garment
> photo into body and sleeve regions, and a deterministic function that
> takes that segmentation plus an instruction like 'front, left chest' and
> works out exactly where that is *on this specific photo* — and refuses to
> place anything if it can't find a safe spot. All under a two-million
> trainable-parameter budget.
>
> I want to walk you through this less like a finished result and more like
> what actually happened — including the dead ends — because I think that's
> the more honest way to show how I think."

`[SHOW: README.md]`

---

## 2. The dataset detour — tell it as a story (2-3 min)

This is the part most people skip in a polished walkthrough. Don't skip it —
it's the strongest evidence of real engineering judgment in the whole
project.

**Say:**

"The ideal dataset, Fashionpedia, already tells us exactly where things like sleeves and collars are. But I couldn't easily get those annotations, so I used another clothing dataset instead. Unfortunately, that dataset doesn't have a sleeve label.

I tried to create my own sleeve labels using the skin labels that the dataset did have. My idea was basically: 'A sleeve is usually close to the person's skin, so I'll look at clothing near the skin and call that a sleeve.'

I tested this idea, but it didn't work. It confused sleeveless clothes with short sleeves, struggled when people crossed their arms, missed some rolled-up sleeves, and generally failed whenever the person's pose or clothing didn't produce a clear skin–sleeve boundary.

So I realized the problem wasn't just that my settings were wrong. The basic assumption — that you can identify sleeves just by looking at what's close to skin — was unreliable. I abandoned that approach."

> "The brief pointed at Fashionpedia, which has real part-level annotations
> — sleeve and collar as their own labeled categories. I tried downloading
> it directly from GitHub first, but the annotations wouldn't come through
> cleanly, only the images. Rather than stall on that, I switched to a
> dataset I could actually get moving on: Kaggle's People Clothing
> Segmentation set. Fifty-nine categories, pixel masks, easy download.
>
> Problem: it has no sleeve category. It has *skin*. So my first idea was —
> grow the skin mask outward a few pixels, and call any garment pixels near
> that boundary 'sleeve.'
>
> I ran it on a batch of images and eyeballed the results. It failed, and it
> failed predictably. A sleeveless top and a short sleeve look identical
> near the shoulder if skin proximity is your only signal. Crossed arms hide
> both underarms. A rolled-up long sleeve puts skin nowhere near the actual
> fabric. And most fashion photography doesn't expose much skin at the
> sleeve boundary in the first place. The core idea was wrong often enough
> that no amount of tuning would have fixed it. I set it aside."

**Pause. Let that land — you're showing you can recognize a dead idea and
kill it, not just push through with something broken.**

> "Next I tried pose estimation instead — a pretrained MediaPipe model to
> find shoulders, elbows, wrists, then draw a tube along the arm and call
> garment pixels inside it 'sleeve.' This worked much better. On thirty test
> images it found sleeves correctly and correctly found nothing on
> sleeveless garments.
>
> I actually hit a real bug here worth mentioning: the first version drew a
> straight line from shoulder to wrist. On a bent arm, that line cuts
> straight through empty space past the elbow and misses the sleeve
> entirely. Switching to a two-segment path — shoulder to elbow, then elbow
> to wrist — fixed it.
>
> But then I made a deliberate call *not* to use this for training labels.
> It's a guess built on top of another model's guess, not ground truth. And
> more fundamentally, I was solving the wrong dataset's problem — Kaggle
> just doesn't have real sleeve annotations, no matter how clever the
> inference. So I went back to Fashionpedia, this time getting both the
> images and annotations to download cleanly, and that pose-estimation work
> didn't go to waste — I reused it later, for placement logic instead of
> training data. I'll come back to that."

`[SHOW: src/prepare_data.py]`

> "From there: Fashionpedia's validation split — about 1,100 images, fully
> labeled, real license — was what I had time to work with rather than the
> full 3GB training set. I kept only images with at least one main garment
> category *and* at least one sleeve annotation present, which left 741
> images. Split 630 train / 111 val, fixed seed, written to a manifest file
> so the split is reproducible and I can't accidentally leak it on a later
> run."

`[POINT TO: prepare_data.py:70-76 — the qualifying_ids filter]`
`[POINT TO: prepare_data.py:54 — target[sleeve] = 2]`

> "One detail worth a beat: where a body annotation and a sleeve annotation
> overlap — right at the shoulder — I let sleeve win. Any ambiguous pixel
> there gets classed as sleeve, not body. That's a conservative choice: I'd
> rather the placement engine treat that pixel as unsafe than falsely trust
> it as safe chest area."

---

## 3. Model choice — where the parameter budget goes (2 min)

`[SHOW: src/model.py]`

**Say:**

> "The brief allowed a compact U-Net from scratch, a small detection model,
> or a frozen pretrained backbone with a trainable decoder. I went with the
> third option, because of where the two-million-parameter budget actually
> goes.
>
> MobileNetV3-Small, pretrained on ImageNet, already understands edges,
> textures, general visual structure before it's seen a single garment
> photo. Frozen weights don't count against the cap — only trainable ones
> do. The backbone is about 927,000 parameters. Freeze all of them, and I
> get that pretrained knowledge for free, and every parameter I *do* train
> goes straight at the actual task: which pixels are body, sleeve,
> background."

`[POINT TO: model.py:73-74 — requires_grad = False]`

> "I built the decoder from scratch. MobileNetV3 processes the image in
> stages, so I tapped three points in that stack — early, where features are
> close to raw pixels; middle, larger shapes; late, abstract and semantic. I
> picked the exact layer indices empirically — ran a dummy tensor through,
> printed shapes at every layer, picked three points with a good spread."

`[POINT TO: model.py:20 — FEATURE_TAPS = {"low": 3, "mid": 6, "high": 12}]`
`[POINT TO: model.py:84-89 — the empirical channel-count probe]`

> "The decoder fuses those three, starting from the most abstract and
> working back to the most detailed — a standard top-down feature pyramid.
> Deep features roughly know *where* the garment is; shallow features know
> *exactly* where an edge falls but not what it means. Fusing them gives
> roughly-correct meaning sitting on precisely-correct boundaries."

### 3a. Decoder architecture — deep dive (have this ready if asked "walk me through the decoder")

`[SHOW: model.py:23-62 — ConvBNAct and LightDecoder]`

**Say, if pressed for the actual architecture:**

> "Concretely: the three taps come off the backbone at 24, 40, and 576
> channels — that last one is wide because it's MobileNetV3's deepest block.
> First step is a 1x1 conv on each tap that projects all three down to a
> common width, 64 channels in the trained checkpoint. That's a deliberate
> bottleneck — a 3x3 conv directly on 576 channels would cost roughly nine
> times what a 3x3 conv on 64 channels costs, and I'm about to run 3x3
> convs, so I shrink the channel count first.
>
> Then it's top-down: upsample the high-level 64-channel map to the mid
> tap's resolution, bilinear, add them together, and pass that through a
> 3x3 conv-BN-ReLU block to smooth out the upsampling artifacts. Repeat one
> more time, fusing that into the low tap's resolution. Then a final 3x3
> smoothing conv and a 1x1 conv down to 3 classes.
>
> Two choices in there are deliberate, not default: bilinear interpolation
> for the feature maps — because these are continuous activations, and
> smoothly blending them between grid points is correct — versus nearest
> interpolation everywhere I resize a *mask*, because a mask is discrete
> class IDs and blending those would average 'body' and 'sleeve' into
> nonsense. Same underlying principle, opposite answer, because the data is
> different. And the smoothing conv after every add — upsample-then-add
> produces blocky seams at the coarse grid boundaries, the conv gives the
> network room to blend that away.
>
> The whole decoder is about 152,000 parameters. I didn't pick 64 channels
> upfront — it started at 32, and I doubled it after the first training run
> showed sleeve badly underperforming, specifically because I had so much
> unused headroom under the 2M cap that doubling it was free to try."

**One correction to make proactively, not defensively, if you're asked about
strides:** the comment in `model.py`'s docstring says the taps are at
"strides 4, 8, 16." I traced the actual shapes through all 13 backbone
layers and the real strides at those tap indices are **8, 16, 32** — the
comment is stale. The tap *choice* and the reasoning behind it don't change;
say "8, 16, 32" if it comes up, and don't be thrown by it — better you
surface it than an interviewer catches it against the code.

### 3b. The frozen-BatchNorm override — deep dive (the subtlest correctness detail in the file)

`[SHOW: model.py:106-112]`

**Say, at whatever depth the room wants:**

> "Every BatchNorm layer holds two different kinds of state. There's the
> learnable scale and shift — gamma and beta — which are normal parameters,
> and since I set `requires_grad = False` on the whole backbone, those are
> already frozen. But BatchNorm also keeps running_mean and running_var as
> buffers, not parameters — and buffers ignore requires_grad completely.
> They update on every training-mode forward pass regardless, as a moving
> average of whatever batch just went through.
>
> PyTorch's `model.train()` flips every submodule into training mode by
> default, frozen or not. So without doing anything else, a 'frozen'
> BatchNorm layer would still normalize using the *current batch's*
> statistics during training, and still drift its running stats away from
> the ImageNet distribution the conv weights were actually calibrated
> against — with a batch size of 8, that drift is especially noisy. At
> inference time, the running stats I load would no longer match what the
> frozen weights expect, and I'd be quietly degrading pretrained features I
> thought I was getting for free.
>
> So I override `.train()` on the model, and for every backbone layer that
> isn't in the explicitly unfrozen tail, I force it into eval mode no
> matter what mode the outer model is in. In eval mode BatchNorm uses the
> fixed running stats it shipped with and never updates them. That's what
> makes the frozen backbone an actually fixed function — not just frozen
> weights, frozen behavior."

`[POINT TO: model.py:106-112]`

> "End result: about 152,000 trainable parameters against a 927,000-parameter
> frozen backbone — comfortably under the 2M cap, with a lot of headroom
> left over. That headroom mattered later."

---

## 4. Training — the honest number progression (3 min)

**This is the section that proves you understand the model, not just that
you ran a script. Use the real numbers.**

**Say:**

> "First version: plain cross-entropy, every pixel weighted equally. Mean
> IoU landed around 0.65 — background 0.96, which you'd expect, body 0.61,
> sleeve only 0.38.
>
> I dug into why sleeve was so weak. Sleeve pixels are a small fraction of
> the image — maybe a quarter of the garment's area, much less of the whole
> photo. A loss that treats every pixel equally barely notices sleeve
> mistakes because there just aren't many sleeve pixels to get wrong. So I
> reweighted the loss by inverse class frequency, and added Dice loss on top
> — Dice rewards region overlap directly instead of treating pixels as
> independent guesses. I also doubled the decoder's width, since I had that
> unused parameter headroom. That combination pushed sleeve to about 0.45,
> body improved a little too, mean IoU to about 0.68."

`[POINT TO: train.py:25-42 — compute_class_weights]`
`[POINT TO: train.py:45-54 — dice_loss]`

> "Then I asked a harder question: was sleeve weak because it's genuinely an
> awkward, thin shape — or was something bigger going on? I measured, per
> validation image, how IoU on each class related to the physical area that
> class occupied. The pattern was clear: body's score depended on garment
> size *even more* than sleeve did. When the garment was small in the frame,
> every foreground class suffered together — the model was working at too
> low a resolution to preserve detail on small or distant garments. I raised
> training resolution from 224 to 320, no other architecture change. Mean
> IoU moved to 0.69, body and sleeve improving together — exactly what the
> size analysis predicted."

> "I tried one more thing: unfreezing the last two backbone layers so a
> small part of the pretrained network could adapt, with a much smaller
> learning rate to protect what it already knew. Body moved up slightly.
> Sleeve came back flat, maybe a hair worse — and the gap between training
> and validation loss widened in exactly the pattern you see when a model
> starts memorizing instead of generalizing. With only 630 training images,
> tripling the trainable parameter count was asking for overfitting. I set
> it aside. It added real complexity and real risk for no real gain — and
> that's the honest finding: sleeve segmentation is genuinely harder given
> the data I had, and the actual fix is more or different training data, not
> more tuning."

**This is a good moment to just say it plainly:**

> "Final numbers: mean IoU 0.693. Background 0.959, body 0.647, sleeve
> 0.473. Sleeve is the weak class and I'm not going to dress that up."

---

## 5. Placement engine — the real bugs, not just the design (3 min)

`[SHOW: src/placement.py]`

**Say:**

> "The core placement idea is simple: subtract sleeve pixels from the body
> prediction, and what's left is strictly torso. Artwork only ever goes
> inside that torso region. Sleeve is never a candidate — not because I
> check for it after placing, but because it's never in the allowed region
> to begin with."

`[POINT TO: placement.py:186 — body_mask & ~sleeve_mask]`

> "The instruction — 'front, left chest' — is parsed for keywords and used
> to carve the torso into a zone. That zone is measured against *this
> photo's* detected torso boundaries, not the whole image — so 'left chest'
> on a close-up and 'left chest' on a distant subject land in totally
> different raw pixel coordinates but both correctly hit the actual left
> chest."

> "Once I have that zone, I don't nudge the artwork into place — I compute
> the point *farthest from any edge of the zone*, using a distance
> transform, and scale the artwork to fit inside that radius. That's not a
> heuristic, it's a geometric guarantee: given the zone, the artwork
> physically cannot cross into a sleeve or off the garment."

`[POINT TO: placement.py:141-148 — find_inscribed_anchor]`

> "Now — three real bugs I found while testing this by hand, because I think
> these are more convincing than the design description on its own."

**Bug 1 — raised arms:**

> "Testing chest-zone logic on photos with a raised arm, I found that
> 'chest' was defined as the top portion of the torso's bounding box. A
> raised arm stretches that bounding box upward toward the hand, which
> quietly shifted 'chest' toward the wrist. I reused the pose model from the
> earlier sleeve experiments — not to guess sleeve shape this time, but to
> find the actual shoulder line and anchor 'chest' to that, regardless of
> arm position."

`[POINT TO: placement.py:114-121]`

**Bug 2 — flat lays:**

> "The segmentation model was trained almost entirely on worn garments. On a
> flat-lay or hanger photo — no person — I actually watched it mislabel a
> coat as mostly 'sleeve.' So I use the *absence* of a detected person as a
> signal that the model's prediction here isn't trustworthy, and fall back
> to a purely geometric estimate off the garment's own outline instead. I
> tested this against a real flat-lay coat photo — the first attempt put the
> logo on the collar, the fallback correctly moved it onto the chest panel."

`[POINT TO: placement.py:179-184]`
`[RUN: python -m src.demo --garment sample_garments/suit.jpg ... if you have time to actually demo it live]`

**Bug 3 — vocabulary gaps and a real failure case:**

> "Testing by hand caught two more things. Someone typed 'left breast'
> instead of 'left chest,' and the code silently treated it as generic
> centered placement, because the keyword list only knew 'chest.' I added
> 'breast' to the vocabulary — but that's the honest reminder that this is
> keyword matching, not language understanding. Anything outside the known
> vocabulary will silently do the wrong thing rather than error.
>
> And I deliberately went hunting for the model's weakest points: a wedding
> dress photo — a style underrepresented in training — got labeled almost
> entirely 'sleeve' instead of body. I checked it against an earlier
> checkpoint to make sure my loss reweighting hadn't caused it — the earlier
> version made the exact same mistake in the same spot. That's a genuine
> training-data coverage gap, not a regression I introduced, and the honest
> fix is more data, not more architecture."

---

## 6. Why a laptop, not a GPU notebook (30 sec, optional — cut if short on time)

**Say:**

> "The brief suggested a free Colab or Kaggle GPU session. I trained on a
> regular laptop CPU instead. With the backbone frozen and a small decoder
> on 630 images, one epoch took under a minute — a full 25-epoch run
> finished in well under half an hour. Setting up and managing a cloud
> notebook would likely have taken as long as the training itself. A GPU
> session was described as sufficient, not required — for a model this
> small, CPU was sufficient too, and every number I'm showing you came from
> code actually run on that machine."

---

## 7. Close with limitations stated plainly, then the one-liner (1 min)

**Say:**

> "To be direct about where this doesn't hold up yet: the instruction parser
> is keyword matching, not real language understanding. The segmentation
> model is unreliable on flat-lay or hanger photos without the pose
> fallback catching it. And a garment style rare in training — like that
> wedding dress — can be confidently misclassified. None of these are
> tuning problems. They're data-coverage problems, and the honest fix is
> more or different training data, not more architecture changes.
>
> If I had to put the whole thing in one sentence: a transfer-learned
> segmentation model that finds body and sleeves in a garment photo, plus a
> deterministic placement engine that uses pose and distance-transform
> geometry to place artwork at the provably safest point in the zone — and
> refuses, with a clear reason, whenever there isn't one."

**Stop talking. Let questions come.**

---

## Anticipated questions and where to point (quick reference)

| If asked... | Point to | One-line answer |
|---|---|---|
| "Why not just fine-tune the whole backbone?" | `model.py:73-74`, the reverted unfreeze experiment | Tried it — overfit on 630 images, sleeve got worse, gap between train/val loss widened |
| "Why sleeve override body at the shoulder?" | `prepare_data.py:54` | Conservative: false-safe body pixel is worse than a false-sleeve pixel |
| "What if segmentation is wrong — does placement still 'guarantee' safety?" | `placement.py:141-148` | Guarantee is *given the predicted region*, not given ground truth — scoped honestly |
| "Why keyword matching, not an LLM/NLP parser?" | `placement.py:72-83` | Deliberately simple for a bounded vocabulary; documented limitation, not hidden |
| "What's the actual failure mode when things go wrong?" | `placement.py:190-213` | Refusal, with a reason string — not a silent bad placement, in most cases |
| "Why CPU?" | `train.py:106` | Small enough model that CPU was faster to set up than a cloud GPU session |
| "Why 1x1 convs before the 3x3s in the decoder?" | `model.py:39-41` | Bottleneck — reduces the 576-channel high tap before the expensive spatial conv |
| "What are the actual tap strides?" | `model.py:94-104` | 8, 16, 32 in practice — the docstring says 4/8/16, that comment is stale, own it |
| "Why does frozen BatchNorm need special handling?" | `model.py:106-112` | Running stats are buffers, not parameters — `requires_grad=False` doesn't stop them drifting; forcing `.eval()` does |
| "How big is the decoder really?" | `configs/train_config.json` (`decoder_channels: 64`) | 152,323 params, confirmed against `outputs/val_metrics.json` |
