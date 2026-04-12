# Tuned LLM Android

A highly optimized Android LLM runner built on `llama.cpp`, designed specifically for high-performance inference on ARM-based mobile devices.

## Core Performance Features

### 1. Optimized ARM Inference
- **OpenBLAS Integration:** Uses OpenBLAS for accelerated linear algebra operations on Android.
- **ARM NEON Repacking:** Full support for ARM NEON weight repacking (`IQ4_NL_4_4`), rearranging weights at load time to match SIMD instruction patterns.
- **Thread Management:** Defaulted to 4 high-performance cores to balance speed and thermal efficiency on Snapdragon/Cortex chipsets.
- **KV Cache Reuse:** Implements stateful conversation tracking. Only new tokens are evaluated between turns, dramatically reducing response latency as history grows.

### 2. The IQ4_NL Advantage
This app is specifically optimized for **Importance Quantization (IQ)** formats. For ARM Android devices, **IQ4_NL** is the recommended standard:

- **Performance:** Benchmarked at **24.4 t/s prompt processing** on Snapdragon 7+ Gen 3 (vs 20.6 t/s for standard Q4_K_M) — an ~18% speedup.
- **Quality:** Uses a non-uniform quantization scheme with optimized lookup tables, preserving more neural network information than uniform Q4_0 at the same file size (3.38GB for Gemma 2B).
- **Efficiency:** Specifically designed to achieve Q4_0 speeds with accuracy levels closer to higher-bit quants.

> **Recommendation:** Use `IQ4_NL` for the best balance of quality and speed on mobile.

## Key App Features

- **Persistent Chat History:** Full sidebar with session management (Create/Delete/Switch) backed by a local Room SQLite database.
- **Model Persistence:** Remembers and auto-reloads your last used GGUF model on startup, preserving the original filename for easy identification.
- **Advanced Copying Support:**
    - **Global Copy:** Export the entire conversation.
    - **Message Copy:** Individual buttons for every chat bubble.
    - **Code Block Copy:** Dedicated copy buttons for every snippet in assistant responses.
    - **Manual Selection:** Full standard text selection enabled for all chat content.
- **UX Optimizations:**
    - **Screen Stay-Awake:** Prevents the display from turning off during active generation.
    - **Markdown Rendering:** Rich text support for assistant answers.
    - **Template Customization:** Adjustable chat templates for different model families (Gemma, Llama, etc.).

## Tech Stack

- **Inference:** `llama.cpp` (C++ JNI Bridge)
- **UI:** Jetpack Compose (Kotlin)
- **Database:** Room (KSP-powered)
- **Build System:** Gradle + CMake (optimized for Termux/Android environments)

## Getting Started

1. Download a compatible GGUF model (e.g., [Gemma 2B IT IQ4_NL](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)).
2. Open the app and select the `.gguf` file.
3. The app will copy it to internal storage for high-speed access and auto-load it on next launch.
