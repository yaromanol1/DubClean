# Third-party notices

DubClean Portable includes or uses the components below. They remain under
their own licenses; the DubClean AGPL license does not replace those terms.

| Component | Use in DubClean | License | Upstream |
|---|---|---|---|
| ClearerVoice-Studio / MossFormer2_SE_48K | speech extraction code and original checkpoint | Apache-2.0 | https://github.com/modelscope/ClearerVoice-Studio |
| MossFormer2_SE_48K fine-tuned checkpoint | DubClean-tuned derivative of the Apache-2.0 base model | Apache-2.0 | base model above; local metadata is stored beside the checkpoint |
| Silero VAD | speech activity detection | MIT | https://github.com/snakers4/silero-vad |
| EfficientAT MN04 AudioSet | singing and music classification on the original English track | MIT | https://github.com/fschmid56/EfficientAT |
| React and ReactDOM | local browser UI runtime | MIT | https://github.com/facebook/react |
| Babel standalone | local browser UI transpilation runtime | MIT | https://github.com/babel/babel |
| hls.js | browser media playback | Apache-2.0 | https://github.com/video-dev/hls.js |
| audio-separator | optional, unbundled song-analysis component | MIT; individual downloaded checkpoints may have separate or unspecified terms | https://github.com/nomadkaraoke/python-audio-separator |
| FFmpeg / ffprobe | external media runtime | license depends on the selected FFmpeg build and enabled components | https://ffmpeg.org/legal.html |

The full Apache License 2.0 text used by the bundled Apache components is in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt). Component-specific
notices bundled with this distribution are:

- [`licenses/React-MIT.txt`](licenses/React-MIT.txt);
- [`licenses/Babel-MIT.txt`](licenses/Babel-MIT.txt);
- [`licenses/Silero-VAD-MIT.txt`](licenses/Silero-VAD-MIT.txt);
- [`licenses/EfficientAT-MIT.txt`](licenses/EfficientAT-MIT.txt);
- [`licenses/Audio-Separator-MIT.txt`](licenses/Audio-Separator-MIT.txt);
- [`licenses/hls.js-Apache-2.0-NOTICE.txt`](licenses/hls.js-Apache-2.0-NOTICE.txt).

Python packages installed by `INSTALL_DEPS.bat`, including Flask, NumPy, SciPy,
SoundFile, PyYAML, PyTorch and their transitive dependencies, retain their
respective licenses. The installed environment's package metadata is the
authoritative inventory for the concrete environment created on a user's
machine.

The optional BS-RoFormer checkpoint named in `portable_manifest.json` is not
bundled or accepted for the production route. Its use requires a separate
license/provenance check for the exact downloaded checkpoint.

Before redistributing a frozen environment or an FFmpeg binary, generate and
review a dependency/license inventory for that exact build. This file is not a
substitute for license notices required by a particular binary distribution.
