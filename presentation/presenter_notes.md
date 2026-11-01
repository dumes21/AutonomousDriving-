# Presenter pack: 15-minute presentation

**Deck:** `RushHour3_Final_Presentation.pptx` (14 slides).
- **Script:** in the speaker notes. Start the slide show in **Presenter View** to read them.
- **Target:** 14:30, leaving 30 s spare.
- **Backup:** `RushHour3_Final_Presentation_backup.pdf` (slides only, no video).

## Run of show

| # | Slide | Speaker | Time | Ends at | Handover (last line) |
|---|---|---|---|---|---|
| 1 | Title and team | All three | 0:45 | 0:45 | Augustus: "…and what we learned." |
| 2 | Business problem | Augustus | 1:00 | 1:45 | "Seth will introduce the environment." |
| 3 | Environment | Seth | 1:00 | 2:45 | "Augustus will give a quick overview of our four approaches." |
| 4 | Four approaches | Augustus | 0:30 | 3:15 | "Seth, over to you." |
| 5 | Imitation learning | Seth | 1:00 | 4:15 | (Seth carries on) |
| 6 | 2D-sim DQN + OpenCV | Seth | 1:15 | 5:30 | "Jia tried to close that gap by learning from images directly." |
| 7 | Image-based DQN in DonkeySim | Jia | 1:30 | 7:00 | "Augustus will now show the approach that worked on the car." |
| 8 | SAC-VAE: seeing vs deciding | Augustus | 1:30 | 8:30 | (Augustus carries on) |
| 9 | Human-in-the-loop training | Augustus | 1:00 | 9:30 | |
| 10 | VAE v1 vs v2 | Augustus | 1:00 | 10:30 | |
| 11 | Results | Augustus | 0:45 | 11:15 | |
| 12 | Demo (83 s clip) | Augustus | 1:45 | 13:00 | "Seth will now take us through the limitations." |
| 13 | Limitations and lessons | Seth | 1:00 | 14:00 | "Jia will wrap up." |
| 14 | Conclusion, then Q&A | Jia | 0:30 | 14:30 | "…we're happy to take questions." |

Speaking time is about 7:30 for Augustus, 4:15 for Seth and 2:00 for Jia, plus the shared title slide. Seth could also take slide 4, which removes one handover.

## Student IDs (the first thing each person says, on slide 1)

- **Augustus:** "I'm Augustus Amedume, student ID 24983037."
- **Seth:** "I'm Seth Yap Wing Shen, student ID 25965486."
- **Jia:** "And I'm Jia Lu, student ID 25404498."

## The demo (slide 12)

- The 83 s clip is embedded in the deck, so no internet is needed.
- It's muted and starts by itself when slide 12 appears. If it doesn't, click it once.
- Talk over it using the notes. If you're running late, click on to the next slide after about 60 s.
- **Settle before the day: which checkpoint, VAE and settings the 30 Oct clip shows.** It's a separate run from the report's test runs, and its laps are slower (see "Likely questions"). Fill in: checkpoint ______ · VAE ______ · throttle range ______.

## If you're running long

- **Slide 4:** cut to 15 s: "Four approaches; only SAC-VAE completed laps on the real car."
- **Slide 6:** drop the training details; keep the 4.6 laps and why the transfer failed.
- **Slide 11:** cut to 15 s and read the three numbers on the right.
- **Slide 12:** move on after about 60 s of the clip.

## Rehearsal (Sun 1 Nov)

- [ ] Run 1 with a stopwatch. Note when each slide ends against the "Ends at" column.
- [ ] Anyone more than 15 s over on a slide trims their notes.
- [ ] Run 2 in Presenter View, with the handovers and the clip.
- [ ] Each person practises the questions below for their own part.

## On the day (Mon 2 Nov)

- [ ] Copy to a USB stick and to OneDrive:
  - the .pptx;
  - the backup PDF;
  - the original clip (`Video.815011750.500498_4050.mp4`).
- [ ] On the room PC:
  - open the deck and start the slide show;
  - check that the clip on slide 12 plays;
  - check that Presenter View shows the notes.
- [ ] If PowerPoint isn't available, present the PDF and play the MP4 on its own.
- [ ] Phones on silent.

## Likely questions

**Why SAC for the final system?** (Augustus)
- It handles continuous steering and throttle directly.
- It's off-policy: every real transition goes into a replay buffer and is reused for many updates, which matters when each sample costs real driving time.
- Its entropy bonus keeps it exploring.

**Why a VAE instead of learning from pixels?** (Augustus)
- The 160×80 colour crop is 38,400 numbers. The VAE compresses it into 32 that capture where the lane is and how it curves.
- Learning to steer from raw pixels would need far more real driving.
- The VAE is trained offline and frozen, so the agent's state doesn't change meaning during RL.

**What exactly is the reward?** (Augustus)
- Each step: 1 + 0.1·(throttle/0.4).
- On operator STOP: −10 − 5·(normalised throttle), and the episode ends.
- Throttle never goes below 0.3, so the car can't earn reward by standing still.

**Why did it fail clockwise?** (Augustus) Three reasons:
- It was trained only anti-clockwise.
- VAE v1 had no clockwise data, so right turns came out as noise (error 485,386).
- The servo's centre was at +0.3, so the car could steer only 0.5 to the right against 1.1 to the left.

**Did VAE v2 fix that?** (Augustus)
- It reconstructs both directions far better, but no policy trained with it was evaluated before the hardware faults. Say so plainly.
- Only if pressed, from our own check (this is not in the report):
  - on frames from the new recording, v2's median reconstruction loss was about 1,960, against about 1.8 million for v1;
  - on the original anti-clockwise recording, v2 was worse: about 30,000 against 2,000.

**How reliable is the driving?** (Augustus)
- Two deterministic test runs: 1,364 steps (37.9 s) and 1,102 steps (30.9 s), about 4.5 laps on average.
- The longest training episode was 3,130 steps.
- We didn't run many repeated trials; that's in our future work.

**Is the demo clip one of those test runs?** (Augustus)
- No. It was recorded on 30 Oct, after the report, and runs longer than either test run.
- Its laps take about 14 s. The report's test runs work out at about 7–8 s per lap (4.5 laps in 31–38 s).
- Explain which checkpoint and settings the clip used (fill in above).

**How much driving did training take?** (Augustus)
- About 26,500 control steps at 22–35 Hz, which is roughly 13–20 minutes of actual driving over six sessions.
- On top of that come the pauses for the 200 updates after each episode.
- This is an estimate from the step count; it isn't stated in the report.

**Why is the VAE training loss negative (report Fig. 13)?** (Augustus)
- The decoder predicts a variance for each pixel as well as a mean, so the loss is a Gaussian negative log-likelihood (`(x−μ)²/σ + log 2πσ`) plus a KL term.
- The log term is negative when the predicted variance is small, so the total can drop below zero. Lower is still better.

**What does the loss on the dashboard mean?** (Augustus)
- It's 0.5·Σ(x−μ)²/σ over the 160×80 crop, the same measure as our VAE viewer notebook.
- A high value means the VAE doesn't recognise the scene.

**Why did imitation learning miss corners?** (Seth) Our likely explanation:
- The model only learned from positions a human drove through.
- Once it drifted off that path on a corner, it had no examples of how to recover.

**Why did the 2D-sim DQN fail on the car?** (Seth)
- The simulator hands the agent exact lane offset and heading. On the car they came from OpenCV, which lost a boundary on corners.
- The policy was very sensitive to observation noise and control delays.

**How did mirroring help in DonkeySim?** (Jia)
- Flipping the image and the steering on half the runs gave the model right-hand bends without a second track.
- Clockwise runs went from 0/3 to 3/3.

**Why wasn't the DonkeySim model tried on the real track?** (Jia)
- The second car's servo was far off-centre, and the Jetson lost power when it steered hard.
- With the camera standing still, the model also switched almost only between hard left and hard right.

**What would you do differently?** (anyone)
- Calibrate the servo first.
- Collect images in both directions and in different light before training the VAE.
- Add a centre-line term to the reward.
- Evaluate over repeated trials.

## Numbers to keep straight (from the final report)

| Topic | Numbers |
|---|---|
| Track | 2.65 m × 1.70 m oval, 0.26 m lane, straights ≈1.00 m |
| Camera | 320×240 RGB; control loop ≈22–35 Hz |
| Imitation learning | 849 frames → 756 kept (608 train / 148 val); ResNet18; L1; AdamW; lr 3e-4; 20 epochs; val MAE 0.0859 |
| 2D-sim DQN | state: offset, heading error, last steering; 3 actions; SB3; lr 1e-3; buffer 100k; 20 min (5.68 M steps); 4.6 laps, 0% crashes over 100 greedy episodes |
| DonkeySim DQN | 160×120 image; 5 actions; stages 150k / 125k / 750k steps; final 11/12 test runs (only failure after 641 steps) |
| VAE | input 160×80 crop → 32 latent; v1: 6,432 images, 100 epochs, right-turn error 485,386; v2: 15,809 images, 62 epochs |
| SAC | state 72 (32 + 20×2); steering ±0.8; throttle 0.3–0.4; steering change ≤0.56/step; γ 0.99; lr 3e-4; batch 64; buffer 30k; 2×32 ReLU; 200 gradient steps per episode; 300 warm-up steps per session |
| Training | 6 sessions; ≈26,500 steps; 200 episodes; mean episode 87 → 343 steps; longest 3,130 |
| Evaluation | 1,364 steps (37.9 s) and 1,102 steps (30.9 s); ≈4.5 laps anti-clockwise; failed clockwise |
| Steering fault | servo centre +0.3; reach 1.1 left vs 0.5 right; fix prepared: offset 0.3, range ±0.7 |
