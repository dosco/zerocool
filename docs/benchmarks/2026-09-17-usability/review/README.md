# Usability review

The second review traced request ownership, streaming completion, failed startup,
client connection lifetime and the evidence checkers. It found and fixed:

- Streaming previously enqueued a successful finish reason before GPU/I/O drain.
  Completion and optional usage now enter the queue together only after drain
  succeeds. A failed drain reports an error and leaves the server failed.
- Chat previously identified its server only by port. It now pins an instance ID
  from the private child handshake or initial explicit connection. A request
  header verifies that identity before reset or inference admission. Replacing a
  server at the same port requires explicit reconnection.
- Failed initialization previously kept its partial executor until server exit.
  The inference worker now drains, clears and destroys it immediately, while the
  failed listener remains observable.
- Cancellation checks now match the exact streaming request ID and enforce the
  500ms acknowledgement gate. A previous request's cancelled flag cannot pass a
  new request's check. Infinite timings and zero/inconsistent physical footprints
  are rejected by comparison helpers.

The regressions cover delayed/failed drain, startup destruction, foreign instance
headers, and a real port replacement while the TUI is connected. Native arithmetic
and experimental execution defaults are unchanged.

`verification.json` records build and binary identities, test counts and hashes of
`raw/` reports. These are model-free checks (plus the real native operator suite),
not full-model API or coding evidence. Full-model usability, paired inference
latency, actual TUI request overhead and the 20-minute coding session remain
unqualified. This review did not retry the previously memory-blocked model run.
