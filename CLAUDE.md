# Claude Instructions

Read [CODING_STYLE.md](CODING_STYLE.md) for naming conventions used in this project.

## Naming Conventions

- Device array variables and kernel parameters pointing to GPU memory must be prefixed with `d_` (e.g., `d_rowIdx`, `d_dirty`, `d_elements`).

## Git

Always create new commits. Never amend existing commits.

When creating a new file, immediately `git add` it.

Never use `git add -A` or `git add .`. Always add specific files by name.

## Repos

- `~/ycjuan.github.io` — my blog repo

## Shortcuts

- **cmc** — "check my comments": fetch and display all unresolved PR review comments on the current branch, then ask the user which ones to act on. Use the GitHub GraphQL API (not REST) because only GraphQL exposes `isResolved` per thread:
  ```bash
  gh api graphql -f query='
  {
    repository(owner: "ycjuan", name: "ycj-blog-code") {
      pullRequest(number: 37) {
        reviewThreads(first: 50) {
          nodes {
            isResolved
            comments(first: 1) {
              nodes { body path originalLine createdAt url }
            }
          }
        }
      }
    }
  }' --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .comments.nodes[0]'
  ```
