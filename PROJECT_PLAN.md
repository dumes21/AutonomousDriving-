# JetRacer Food Delivery — Project Plan

**Subject:** 43008 Reinforcement Learning, Assignment 3 (group project)
**Team:** Rush Hour 3 (Seth Yap Wing Shen, Jia Lu, Augustus Amedume)
**Hard deadline:** Fri 23 Oct 2026 (Week 12) · **Team target:** submit Thu 22 Oct
**Plan written:** Wed 16 Sep 2026

---

## 1. Context for Claude Code

- **Goal:** an RL agent that drives an NVIDIA JetRacer (Jetson Nano) around a track using its camera, framed as a delivery task.
- **"Autonomous driving"** in this subject means completing laps around a track. Completing the marker's track is the basic requirement.
- **Delivery task:** two coloured paper patches on the track. The car stops on the **blue pickup** patch, then drives to the **orange drop-off** patch and stops there.
- **Reference codebase:** https://github.com/masato-ka/airc-rl-agent (VAE + SAC; supports DonkeySim and the real JetRacer).
- **Simulator:** DonkeySim via `gym-donkeycar`.
- **Staged approach (from our Part B proposal):** sim first, then deploy to the physical JetRacer.

### Suggested repo layout

```
jetracer-delivery/
├── README.md
├── PROJECT_PLAN.md          # this file
├── configs/                 # hyperparameters, HSV thresholds, reward weights
├── sim/                     # DonkeySim env wrappers, virtual patches
├── vision/                  # HSV patch detector, frame preprocessing
├── models/                  # VAE training + checkpoints (gitignored weights)
├── rl/                      # SAC training, delivery reward, observation builder
├── car/                     # JetRacer control, safety stop, teleop, deploy script
├── eval/                    # evaluation runs, metrics logging, plots
└── report/                  # figures and notes for the final report
```

---

## 2. Scope (priority order)

### Must — the pass
- [ ] RL policy completes laps of the marker's track from camera input
- [ ] Safety stop when the track is lost or camera frames are bad
- [ ] Metrics: laps without intervention, lap time

### Should — our pitch
- [ ] Stop on blue pickup patch, continue, stop on orange drop-off patch
- [ ] Delivery success rate over 10+ runs
- [ ] Sim vs real comparison

### Could — stretch
- [ ] Two drop-off colours; target assigned at episode start
- [ ] Wrong-destination rate
- [ ] Test on a second track layout

---

## 3. Technical approach

### Pipeline

```
camera frame ──► VAE encoder ──► latent z (~32 dims)
                                      │
             delivery features ───────┤   (patch_seen, patch_colour, stage, target)
                                      ▼
                               SAC policy ──► steering, throttle (continuous)
```

### Components

- **VAE:** compresses cropped/resized camera frames to a small latent vector. Train one on sim frames and one on real frames.
- **SAC:** continuous-action RL for steering and throttle, on top of the latent plus delivery features. Include a short action history as in airc-rl-agent.
- **Patch detector:** HSV colour threshold on the bottom region of the frame. Not learned. Require the colour in several consecutive frames before counting it.
- **Stage logic:** `to_pickup` → `to_dropoff` once the car has stopped on the pickup patch for ~2 s.

### Reward

| Component | Sign |
|---|---|
| Progress along track | + |
| Staying near lane centre | + |
| Stopping on the correct patch | + bonus |
| Leaving the track | − (terminates episode) |
| Stopping off-patch | − |
| Driving past the target patch | − |

### Sim → real

- DonkeySim can't easily render paper patches. In sim, use **virtual patches**: fixed track positions read from sim telemetry (`info` position / progress).
- On the car: retrain the VAE on real frames, then fine-tune or retrain SAC in short sessions with a person resetting the car.

### Fallback

If learned stopping won't converge: **RL policy drives, patch detector triggers the stop.** Present it as a hierarchical design and report learned stopping as a limitation.

---

## 4. Week-by-week plan

### Week 1 — Setup (16–20 Sep)
**Sim & RL**
- [ ] Create shared repo and environment (conda/venv, requirements file)
- [ ] Install `gym-donkeycar` and DonkeySim on a GPU laptop
- [ ] Run the airc-rl-agent sim demo end to end

**Car & vision**
- [ ] Confirm JetRacer arrival date with tutor
- [ ] Buy tape, blue + orange matte card, spare batteries
- [ ] Mark out a practice track

**Report**
- [ ] Get the final report rubric; build the report outline from `Project-FinalReport-Template-Part-E.docx`
- [ ] Assign section owners

**Milestone:** sim demo runs on at least one laptop.

### Week 2 — Laps in sim (21–27 Sep)
**Sim & RL**
- [ ] Collect sim frames
- [ ] Train the VAE on sim frames
- [ ] Train SAC for lane following
- [ ] Log reward curves from the first run (TensorBoard or CSV)

**Car & vision**
- [ ] Assemble car, flash Jetson image
- [ ] Test camera, steering, throttle; drive manually (teleop)
- [ ] Measure a safe max throttle

**Report**
- [ ] Write MDP formulation (state, action, reward, termination)
- [ ] Write environment description

**Milestone:** 3 consecutive laps in sim · car drives manually.

### Week 3 — Delivery in sim (28 Sep – 4 Oct)
**Sim & RL**
- [ ] Add virtual patches, stage flag and target to the observation
- [ ] Implement and tune the delivery reward
- [ ] Train the sim delivery agent

**Car & vision**
- [ ] Record real track frames under varied lighting
- [ ] Train a real-world VAE
- [ ] Build and tune HSV patch detector; measure accuracy

**Report**
- [ ] Method section: VAE + SAC, why SAC suits continuous control
- [ ] Background / literature

**Milestone:** sim agent completes a delivery · detector ≥ 95% accurate on patches.

### Week 4 — Real-car laps (5–11 Oct)
**Sim & RL**
- [ ] Support real-car training: hyperparameters, checkpointing
- [ ] Sim baseline evaluation (10 runs per configuration)

**Car & vision**
- [ ] Train SAC on the real car for lane following
- [ ] Implement safety stop (lost track / invalid frames)
- [ ] **Tag working version `v1-laps`**

**Report**
- [ ] Draft sim results section with figures

**Milestone:** real car laps unassisted → basic requirement secured.

### Week 5 — Delivery & evaluation (12–18 Oct)
**Sim & RL**
- [ ] Stretch: assigned destination with two drop-off colours
- [ ] Sim vs real comparison analysis

**Car & vision**
- [ ] Enable pickup → drop-off stages on the car
- [ ] Evaluation: 10+ runs per configuration
- [ ] Test on a changed track layout
- [ ] Film backup demo runs

**Report**
- [ ] **Full draft by Sun 18 Oct** (results, discussion, limitations)

**Milestone:** full report draft · backup demo video recorded.

### Week 6 — Freeze & submit (19–23 Oct)
**Sim & RL**
- [ ] Code freeze Mon 19 Oct
- [ ] README with reproduction instructions

**Car & vision**
- [ ] Demo rehearsal, charged batteries, spare patches
- [ ] Bug fixes only

**Report**
- [ ] Edit, check against rubric, record individual contributions
- [ ] **Submit Thu 22 Oct**

---

## 5. Evaluation metrics

| Metric | Definition | Where | Tier |
|---|---|---|---|
| Laps without intervention | Laps completed before a human touches the car, per run | Sim + real | Must |
| Lane-centre error | Mean \|cross-track error\| (sim telemetry; camera estimate on car) | Sim + real | Must |
| Episode reward | Learning curve, mean ± std across seeds | Sim | Must |
| Delivery success rate | % of runs stopping on pickup then correct drop-off | Sim + real | Should |
| Delivery time | Seconds from start to drop-off stop | Sim + real | Should |
| Wrong-stop rate | Stops off-patch or at wrong destination, per run | Sim + real | Could |

---

## 6. Risks

| Risk | Impact | Response |
|---|---|---|
| JetRacer arrives late | High | Sim work doesn't depend on it. If car isn't running by 4 Oct, drop the Could tier and do delivery in sim only. |
| Marker's track differs from ours | High | Train VAE on varied layouts, tape colours, lighting. Test on a changed layout in week 5. |
| Real-car SAC training unstable | High | Low throttle cap, short sessions, frequent checkpoints. Fallback: pretrain from teleop data, then RL fine-tune. |
| Learned stopping won't converge | Medium | Detector-triggered stop (hierarchical design); report as limitation. |
| Lighting breaks colour detector | Medium | Matte card, calibrate HSV at demo venue, multi-frame confirmation. |
| Report left too late | Medium | One section per week; full draft due 18 Oct. |

---

## 7. Roles (starting split — swap to match strengths)

| Role | Responsibilities | Report sections |
|---|---|---|
| Sim & RL training | DonkeySim, VAE, SAC, reward design, learning curves | MDP, method |
| Car & deployment | Assembly, Jetson setup, safety stop, real-car training, demo | Platform, deployment |
| Vision, evaluation & report lead | HSV detector, delivery stages, evaluation protocol, results | Results; edits whole report |

Everyone does at least one real-car training session.

---

## 8. Assumptions

- Week 12 ends Fri 23 Oct 2026, with no teaching break before then.
- Report structure follows the Part E final report template/rubric — check it in week 1.
- JetRacer order (~10 days) arrives by late September.
