#!/bin/sh
# Point this clone at the versioned hooks. Worktrees share .git, so one run
# covers every worktree of the clone.
set -eu
root=$(CDPATH= git rev-parse --show-toplevel)
git -C "$root" config core.hooksPath scripts/git-hooks
echo "hooks: core.hooksPath=scripts/git-hooks"
