# Joint static Q5 formal deployment

This directory is the persistent Nano222 configuration for the validated
traffic plus industrial static Q5 release.

- Edge API: `0.0.0.0:18101`
- llama-server: `127.0.0.1:18190`
- Cloud API: `http://192.168.31.135:18100`
- Release: `traffic-industrial-joint-static-v2-q5km`
- Model SHA-256: `308daa980c7ca295e18bd76e8dcf6dc1ed725ded32ada535a0c5c1910c695ce2`
- Runtime adapters: none
- Scene prompts: traffic `T` and industrial `I`, both 17 input tokens and
  one constrained output token

The formal release store and all mutable edge databases live under
`runtime/traffic_industrial_joint_static_q5km_formal_v1`. The old Q6 service
is retained as the named rollback target and must remain disabled while this
unit is active. The obsolete `cloud-edge-edge-18101-5f488235.service` must
also remain disabled to prevent reboot-time topology drift.

The unit deliberately uses default mmap, CUDA graphs enabled, context 128,
batch/ubatch 16, one slot, and all layers on GPU. It must not contain any
`--lora`, `--lora-scaled`, `--no-mmap`, or CUDA-graph-disable option.
The industrial provider timeout is 500 ms so the second member of a concurrent
RGB/infrared review pair can wait for the validated single slot. The formal
end-to-end SLA gate is the fixed 180-event population mean, not its maximum.

Formal measurements on 2026-08-13:

- Industrial fixed population: 180/180 compact events, local accuracy 1.0,
  Qwen selected/completed 60/60, rule agreement 1.0, authoritative-final rate
  1.0, and input-to-provisional mean 111.077345 ms. The raw evidence SHA-256
  is `c260da971fe5be5b64fe1d7b7bbd88d67a4c6a0342b864962bb6f21307779333`.
- Traffic continuous window: 100 samples/400 events, 400 successful routes,
  complete aggregation for all 100 samples, and event business-completion mean
  174.240621 ms. Qwen was selected 30 times with zero runtime errors, but all
  30 predictions failed candidate-action authorization and safely fell back to
  the student, so the strict Qwen-accepted gate remains failed. The raw
  evidence SHA-256 is
  `2d138097e7f846fa297d6f4d7a742c2fa27f03254c07419f382838c6daa84988`.

The traffic acceptance result must not be reported as passed. The next model
optimization is authorization-aware traffic training or decoding, not a
relaxation of the action safety contract.

Rollback:

```bash
bash scenes/industrial_anomaly/deployment/joint_static_formal_v1/restore_q6.sh
```

The rollback changes the systemd unit as well as the model topology. A
release-registry pointer rollback alone is not sufficient.
