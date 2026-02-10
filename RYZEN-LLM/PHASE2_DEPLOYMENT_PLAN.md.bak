# Phase 2 Deployment Plan — Memory Pool + Threading

## Purpose

Deploy Phase 2 (memory pool, threading, AVX2 kernels) to production safely and measurably.

## Summary

Phase 2 achieved **56.62 tok/s**, exceeding the 15–25 tok/s target. This plan covers steps to merge, build, stage, and deploy with clear rollback criteria and monitoring.

---

## Pre-deployment Checklist (must be completed)

- [ ] All unit, integration, and performance tests pass locally and in CI
- [ ] `PHASE2_RESULTS.md` and `PHASE1_PHASE2_PROGRESSION.txt` are up-to-date and reviewed
- [ ] Change log / release notes drafted and approved
- [ ] Deployment owner and on-call person assigned (name, contact)
- [ ] Backup snapshot of current production model/artifacts (retain for quick rollback)
- [ ] Staging environment ready (matching prod config, same CPU profile where feasible)

## Release Artifacts

- Source branch: `main` (or `release/phase2` if branching strategy requires)
- Tag: `v2.0-phase2` (example)
- Binaries: `ryzen-llm.tar.gz` or Docker images `ryzen-llm:phase2` pushed to registry
- Release notes: `CHANGELOG.md` and `PHASE2_RESULTS.md`

## Deployment Steps

1. Finalize and commit docs: `PHASE2_RESULTS.md`, `PHASE2_DEPLOYMENT_PLAN.md`, `CHANGELOG.md`.
2. Run full CI pipeline (unit, integration, performance). If CI fails, **stop** and triage.
3. Create release branch `release/phase2` and open PR (if required); request approvals.
4. Merge to `main` after approvals, tag `v2.0-phase2`.
5. Build release artifacts (binaries and Docker images). Publish to registry.
6. Deploy to **staging** environment:
   - Run smoke tests: basic RPCs, small-scale inference, perf smoke
   - Run performance verification: reproduce Phase 2 benchmark (single-node baseline)
   - Validate correctness against reference outputs
7. If staging checks pass, proceed to canary rollout (if supported) or full production rollout:
   - Canary: route 5–10% traffic to new release for 30–60 minutes
   - Monitor CPU, memory, latency, throughput, error rates, and custom SLOs
8. Promote to full production after canary success.

## Rollback Plan

- If smoke tests or monitoring show issues during staging or canary:
  1. Immediately stop rollout and revert traffic to previous release
  2. Redeploy known-good tag (`v1.x`) and confirm health
  3. Create an incident with logs, captured metrics, and rollback timeline

## Verification & Acceptance Criteria

- Unit and integration tests: **PASS** in CI
- Throughput: >= 50 tok/s in staging synthetic load (target: 56.62 tok/s or within ±5%)
- Latency/99p: within expected range compared to baseline
- No correctness regressions (validation dataset checksums match)
- No increase in error rates or crash reports for 60 minutes post-promote

## Monitoring & Metrics

- Dashboard: throughput, CPU%, memory, 99th latency, error rate
- Alerting: SLO breach, CPU saturation > 95% for > 5 minutes, errors > baseline + 2σ
- Owner: on-call engineer (name/email), fallback: engineering lead

## Post-Deployment

- Collect final benchmark and confirm numbers in `PHASE2_RESULTS.md`
- Add release notes with observed metrics
- Schedule retrospective if any incidents occurred
- Archive release artifacts and tag them in project storage

## Contacts & Roles

- Deployment owner: @owner (assign actual person)
- On-call/monitoring: @oncall
- Reviewers: @team-leads

---

## Scripted Commands (examples)

```powershell
# Local build + tests (MSVC):
.
# (Example) Build and run tests
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build --output-on-failure

# Tagging and push
git checkout main
git pull origin main
git checkout -b release/phase2
# update changelog
git commit -am "chore(release): Phase 2 deployment" && git push origin release/phase2
# Create PR, get approvals, then merge

# Build Docker image
docker build -t registry.example.com/ryzen-llm:phase2 .
docker push registry.example.com/ryzen-llm:phase2
```

---

## Notes

- Keep the canary window conservative (30–60 minutes) and have rollback commands ready.
- Verify performance on representative hardware where possible.
- If production workload differs significantly, prefer staged canary instead of immediate full rollout.

---

_Created automatically as part of the Phase 2 deployment process._
