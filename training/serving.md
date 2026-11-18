# Serving note — how the fine-tuned model would be served (DRAFT)

Short note only. **No serving code, no deploy.** The fine-tuned models
do not exist yet; this describes the intended serving path so the
training choices (CTC vs Whisper) stay aligned with it.

## Which model goes where

| Use case | Model | Why |
| --- | --- | --- |
| Phone-call assistant (live calls) | **CTC (XLS-R 300M fine-tune)** — default | Latency (see below) |
| Batch / offline transcription, best accuracy | Whisper-LoRA (large-v3-turbo) | Higher accuracy on clean read speech per the 2026 baseline; latency is affordable offline |

## Why CTC is the default for the phone-call use case

A phone call is a latency budget, not an accuracy contest: audio
arrives in real time, the user is waiting on the line, and a reply
that arrives after a long pause is a failed interaction even if the
transcript is perfect.

- **CTC decoding is non-autoregressive.** A Wav2Vec2-CTC model emits
  the whole transcription in one forward pass plus a cheap greedy
  (or beam) decode over the output logits. There is no token-by-token
  generation loop.
- **Whisper decoding is autoregressive.** The decoder generates one
  token at a time, each step attending over the full encoder output,
  until it emits end-of-text — for a 10-second utterance that is
  dozens of sequential decoder steps, and it cannot start producing
  the tail of the transcript until the earlier tokens exist.
- The 2026 benchmark (Paudel & Sayami) measured the difference
  directly: **CTC decoders ran ~29× faster than autoregressive
  Whisper in real-time factor** on the same hardware. That is the
  number this choice rests on — it is theirs, quoted with attribution,
  not a measurement of our models (which do not exist yet).
- On a phone call the practical consequence is streaming: a CTC model
  can emit partial transcripts as audio arrives (frame-synchronous),
  while Whisper needs the whole segment before it can decode it.
  Partial transcripts are what make barge-in, live captions, and
  low-latency replies possible.

Accuracy trade-off, stated plainly: on the 2026 benchmark the two
families were close on clean read speech (Whisper-Large-v3-Turbo
14.76% WER in-domain vs IndicWav2Vec 14.89%), so giving up Whisper's
small accuracy edge for CTC's large latency win is the right trade
for a live call. On noisy/phone audio neither family's numbers are
known — the realistic slices are empty until Phase 2 data exists —
so the latency argument, not an accuracy claim, is the reason CTC is
the default.

## How each model would be served

### CTC model — CTranslate2 or ONNX

- **CTranslate2**: quantise the fine-tuned Wav2Vec2-CTC model
  (int8) and serve it with CTranslate2's inference engine. Small
  footprint, fast CPU inference, streaming-friendly. This is the
  preferred path for a low-cost CPU server or an on-device deployment.
- **ONNX**: export the model to ONNX (`torch.onnx` / Optimum) and
  serve with ONNX Runtime. Portable across runtimes, easy to
  quantise, and the same artifact can run on CPU or GPU. Use this if
  the deployment target already runs ONNX Runtime, or if CTranslate2's
  Wav2Vec2 support is a problem at export time.
- Either way, the served artifact is the **fine-tuned checkpoint**
  (model + processor), greedy-decoded by default; a beam search with
  a Nepali language model is a later optimisation, not part of the
  first serving step.

### Whisper model — faster-whisper

- **faster-whisper** (CTranslate2-backed Whisper): load the base
  `openai/whisper-large-v3-turbo` plus the LoRA adapter — note that
  the adapter must be **merged into the base weights first** (LoRA
  adapters are not served standalone; merge with PEFT's
  `merge_and_unload()` and export the merged model), then run it
  through faster-whisper. It is 4–5× faster than the reference
  `openai-whisper` implementation on the same hardware, with int8
  quantisation available — but it is still autoregressive, so it does
  not close the latency gap with CTC for live calls.
- Use faster-whisper for batch jobs (transcribing collected audio,
  building the Phase 2 test set) and for any accuracy-first endpoint
  where a multi-second delay is acceptable.

## What is NOT decided here

- Hosting (no deploy in this phase; Render free tier is suspended
  until 1 Dec 2026 in any case — see the repo's Phase 4 notes).
- Streaming protocol (WebSocket vs chunked HTTP) for the phone-call
  path — that belongs to the phone-call assistant's design, not to
  model serving.
- Whether the CTC model needs a language model in the decoder — to be
  decided from the per-clip error analysis after the first real run.
- Quantisation accuracy: int8 quantisation of either model must be
  re-scored on the v0 holdout before it ships; a quantised model is a
  different model until proven otherwise.
