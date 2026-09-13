# Eight-launch native HTTP campaign, September 13, 2026

These are exact-byte copies of the completed controller, short/long counting
matrices, code-review streams, HTTP checks and seeded samples for all eight
launches. Order: all / last / vec4 / combined / combined / vec4 / last / all.
Each long stream has 16384 input and 512 output tokens; short matrices use
128/512/2048/8192 input and 128 output. Every request starts with fresh KV.

See the [report](../RESULTS-halo-combined-profile.md) and
[complete corrected audit](../results-halo-combined-http-audit.json).
The configuration/build identities are embedded in each raw matrix header.
[Provenance](provenance.json) binds archived files and the private full
telemetry, server logs, controller logs and frozen input manifests.

The original complete audit failed because the observer stopped sampling
at the normal stop-sigterm transition on launches 6/7. The original controller
error and last-only fallback decision are preserved, not overwritten.
Separate stop-evidence records contain the successful contemporaneous
systemd stop jobs and hash-bound completed-work samples. The corrected audit
requires these records for a transitional final sample; missing or corrupt
evidence fails. Numerical/performance/memory limits were not relaxed.
decision-recovered.json applies the original selection function and selects
combined. recovery-v2.json binds that correction to the original failure.

Full telemetry remains in /home/ryan/qk-combined-profile-kGqp5a on Max.
The current offline auditor expects that complete private bundle and the
original shared fixture/reference paths; this smaller repository archive
alone is not a portable rerun environment. Reproduce live tests only in a
fresh output directory with the exclusive-GPU and memory guards; do not
restart completed controllers or overwrite evidence.
