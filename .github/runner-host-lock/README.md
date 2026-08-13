# Shared runner host lock

These hooks serialize GitHub Actions jobs across multiple runner registrations on
the same small host. The start hook acquires a host-wide lease before any workflow
step runs; the completion hook releases it after all steps finish.

The lease records the runner worker's PID, process start identity, and host boot
identity. A later job can therefore reclaim the lock after a worker crash or host
reboot without mistaking a reused PID for the original owner.

Install this directory outside every Actions runner application directory, for
example at `/home/pipeline/actions-runner-hooks`. Configure each runner's `.env`:

```text
HOST_LOCK_RUNNER_ID=<unique-runner-name>
ACTIONS_RUNNER_HOOK_JOB_STARTED=/home/pipeline/actions-runner-hooks/acquire.sh
ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/home/pipeline/actions-runner-hooks/release.sh
```

Restart every runner after editing `.env`. Optional settings are
`HOST_LOCK_ROOT`, `HOST_LOCK_POLL_SECONDS`, and `HOST_LOCK_TIMEOUT_SECONDS`; the
defaults are `/home/pipeline/actions-runner-host-lock`, 2 seconds, and 6 hours.

GitHub assigns a job before the start hook runs, so a job waiting for the shared
lock appears assigned or in progress in the Actions UI rather than queued.
