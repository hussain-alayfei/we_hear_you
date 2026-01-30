# Safe Chat 2.0: Optimization Summary

This document captures the successful architecture and settings for the Sign-to-Text "Safe Chat" system as of Jan 2026.

## 1. Hybrid Architecture (The "Fix")
We moved from a purely server-side approach to a **Hybrid Model** to eliminate lag.

*   **Visualization (Client-Side)**:
    *   Uses **MediaPipe `handLandmarker`** directly in the browser (`detectForVideo`).
    *   **Result**: Instant 60fps skeleton tracking with 0ms latency. The user sees their hand move perfectly smoothly.
*   **Prediction (Server-Side)**:
    *   Uses a **Throttled API Call** to the Flask server (`/predict`).
    *   **Result**: Heavy AI processing happens in the background without freezing the video/skeleton.

## 2. Speed & Stability Settings (Golden Balance)
The system is tuned to balance speed with accuracy using the following constants in `index.html`:

```javascript
// Current "Stable Balance"
const SAFE_PREDICTION_INTERVAL = 150; // Check hand every 150ms (~6.6 times/sec)
const STABILITY_THRESHOLD = 7;        // Hold for 7 matches (~1.05s) to confirm letter
const MIN_TIME_BETWEEN_LETTERS = 1000; // Wait 1s before typing next letter
```

*   **Logic**:
    1.  User signs a letter.
    2.  System waits for ~1 second of consistent detection (`STABILITY_THRESHOLD`).
    3.  Letter is typed.
    4.  System enforces a 0.8s cooldown (`MIN_TIME_BETWEEN_LETTERS`) to allow hand transition.

## 3. Typing Flow
*   **No Locks**: We removed the "One Letter Per Session" lock. You can type the same letter twice (e.g., "Allah") by holding, waiting for the cooldown, and holding again.
*   **Word Separation**: The **Space (Confirm)** button is used to finalize a word and insert a space. This is critical for Arabic letter shaping (connected forms).
*   **Send**: The **Send** button moves the drafted text to the chat bubble.

## 4. UI Fixes (RTL)
*   **Direction**: `dir="rtl"` applied to Chat input areas.
*   **Input Bar**: Swapped layout -> **[Input Field] [Send Button]** (Send is on the Left/End).
*   **Drafting Bar**: Text aligns to the Right (`justify-start` in RTL flex).

## 5. Key Code Block (`processSafeChatFrame`)
This is the core loop that makes it fast:

```javascript
async function processSafeChatFrame(video, canvas, ctx) {
    if (!safeChatMode) return;

    // 1. Draw Background
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Server Prediction (THROTTLED)
    const now = Date.now();
    if (now - lastSafeChatPredictionTime > SAFE_PREDICTION_INTERVAL) {
        // Send clean frame to server...
    }

    // 3. Client-Side Skeleton (INSTANT)
    if (handLandmarker) {
        const results = handLandmarker.detectForVideo(video, performance.now());
        if (results.landmarks) {
             drawHandLandmarks(results.landmarks, ...);
        }
    }
    
    requestAnimationFrame(...);
}
```
