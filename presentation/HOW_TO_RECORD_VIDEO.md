# Recording the 3–5 minute project video

**Due:** Mon 2 Nov 2026, 11:59 PM · **Format:** MP4, 3:00–5:00 long · **Target:** 4:30

Record with `RushHour3_Video_Deck.pptx` (8 slides). Everything except your voices is already in it:

- Each slide is timed to the plan below. Recording your narration replaces these timings with your own.
- The demo clip on slide 6 starts by itself, is muted and is trimmed to its first 45 s. Only your voices will be heard.
- The script is in each slide's speaker notes. It is also copied at the end of this file so you can read it on your phone.

## Who says what

| Slide | Time | Section | Voice |
|---|---|---|---|
| 1 | 0:00–0:20 | Title: name and student ID | Augustus, Seth, Jia |
| 2 | 0:20–1:20 | Introduction and background | Augustus → Seth → Jia → Augustus |
| 3 | 1:20–2:20 | Business problem | Augustus |
| 4 | 2:20–2:50 | Solution: SAC-VAE | Augustus |
| 5 | 2:50–3:10 | How it learned | Augustus |
| 6 | 3:10–3:55 | Demo (45 s clip) | Augustus, talking over the clip |
| 7 | 3:55–4:10 | Results | Augustus |
| 8 | 4:10–4:30 | Limitations, next steps, thanks | Jia |

Slides 1 and 2 need all three voices, so record together: one laptop, one quiet room.

## Before you record (5 minutes)

1. Open the deck in desktop PowerPoint, not PowerPoint for the web.
2. Turn off notifications and close anything that might make a sound.
3. Read the script aloud once with a stopwatch. A few seconds either side of each slide's time is fine.
4. Record a 10-second test on slide 1, play it back and check the volume. Sit about an arm's length from the laptop microphone.

## Record

1. **Record** tab → **From Beginning**. In older versions: **Slide Show → Record Slide Show → Start Recording from Beginning**, with *Narrations* and *Slide and animation timings* ticked.
2. To see the script while you record, use the **Notes** (or **Teleprompter**) button at the top of the recording window.
3. Press the red **Record** button, wait for the countdown, then speak.
4. Press **→** for the next slide when that slide's lines are done. PowerPoint doesn't record sound during a slide change, so pause briefly before and after each one.
5. On slide 6 the clip plays for 45 s. Keep talking over it, and press **→** only once it has finished.
6. Press **Stop** after slide 8.
7. To redo one slide: go to it, choose **Clear → Clear Recording on Current Slide**, then **Record → From Current Slide**.

## Export the MP4

1. **File → Export → Create a Video**.
2. Choose **Full HD (1080p)** and **Use Recorded Timings and Narrations**.
3. Click **Create Video** and save as `RushHour3_Video.mp4`. Wait for the progress bar at the bottom of the PowerPoint window to finish (a few minutes).
4. Watch the whole file once and check that:
   - it runs between 3:00 and 5:00;
   - all three voices and student IDs are clear;
   - the clip plays on slide 6.
5. If the upload rejects the file size, export again at **HD (720p)**.
6. Submit before 11:59 PM on Mon 2 Nov.

## Script

The same text is in the slide notes. If you change the wording, change it in the notes too.

**Slide 1** · VOICES: all three · TITLE · 0:00–0:20 (20 s)

> **Augustus:** Hi, we're team RushHour3, project 75. I'm Augustus Amedume, student ID 24983037.
>
> **Seth:** I'm Seth Yap Wing Shen, student ID 25965486.
>
> **Jia:** And I'm Jia Lu, student ID 25404498.

**Slide 2** · VOICES: Augustus, Seth, Jia · INTRODUCTION & BACKGROUND · 0:20–1:20 (60 s)

> **Augustus:** In this subject, almost everything we trained lived in a simulator. We wanted to know what happens when reinforcement learning has to drive real hardware. Our platform is the NVIDIA JetRacer, a small car with a Jetson Nano computer and a front camera, on an oval track with a lane just 26 centimetres wide. We tried four approaches.
>
> **Seth:** I built imitation learning, which copies a human driver, and a DQN trained in a 2D simulator and run through OpenCV lane detection. Both looked good offline but struggled on the real car.
>
> **Jia:** I trained an image-based DQN in a DonkeySim copy of our track. It passed 11 of 12 simulator tests, but real camera images looked too different.
>
> **Augustus:** Our fourth approach, SAC with a variational autoencoder, learned on the car itself. That's the one we'll show you.

**Slide 3** · VOICE: Augustus · BUSINESS PROBLEM · 1:20–2:20 (60 s)

> Why does this matter? Delivery services still depend on human drivers to pick up and drop off goods on time. That brings driver fatigue, high operating costs and inconsistent performance. Autonomous delivery vehicles could take over the repetitive driving.
>
> But a delivery vehicle has to cope with the real world, not a simulator: its camera is noisy, the light changes, and its steering is never perfect. Our project is a small-scale proof of concept of that problem: a JetRacer acting as a delivery vehicle. It must follow its lane using only its camera, take the corners on its own, and, as the final goal, stop at pickup and drop-off zones.
>
> So our question was: can a real car learn reliable lane following from camera images using reinforcement learning?

**Slide 4** · VOICE: Augustus · SOLUTION · 2:20–2:50 (30 s)

> Our solution separates seeing from deciding. A variational autoencoder, trained offline on our own track images, compresses each camera frame into 32 numbers. Together with the last 20 actions, that's the state. Soft Actor-Critic then learns which steering and throttle to send. Because SAC is off-policy, every real-world sample is reused many times, which makes learning on a physical car practical.

**Slide 5** · VOICE: Augustus · HOW IT LEARNED · 2:50–3:10 (20 s)

> All the learning happened on the real car. We start an episode, the car drives itself, and we press stop when it leaves the track. That step is penalised, and SAC trains while we reset the car. In total: six sessions and about 26,500 steps.

**Slide 6** · VOICE: Augustus · DEMO · 3:10–3:55 (45 s; the clip plays muted, trimmed to 0:00–0:45)

> Here's the trained policy driving by itself. Nobody is steering; the operator only watches, ready to press stop. Notice the small corrections through each corner. The car keeps going, lap after lap, anti-clockwise, without leaving the lane.
>
> Behind the scenes, the car only sees the bottom of each camera frame, squeezed into 32 numbers by the VAE, and the policy answers with a new steering and throttle command more than twenty times a second.

**Slide 7** · VOICE: Augustus · RESULTS · 3:55–4:10 (15 s)

> During training, mean episode length grew from 87 to 343 steps. In evaluation, the final policy averaged about 4.5 laps anti-clockwise. Clockwise, it still failed on right-hand curves.

**Slide 8** · VOICE: Jia · CLOSE · 4:10–4:30 (20 s)

> Our main limits were the sim-to-real gap, an off-centre steering servo, and training in one direction only. Next, we'd train both directions with our improved VAE and add pickup and drop-off stops. Thanks for watching!
