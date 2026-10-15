# Upstream drafts

Drafts of what our fork could offer upstream. Nothing here is posted: each needs the user's OK,
and pull requests to LiteRT and ML Drift need Google's CLA signed first. Links to our forks work
once the branches are pushed.

How each project takes contributions (checked 2026-10-15):

| Project | Takes | Notes |
| --- | --- | --- |
| `google-ai-edge/LiteRT-LM` | issues | Not accepting code; merged PRs come from Google's internal mirror |
| `google-ai-edge/LiteRT` | pull requests | Individual or corporate CLA |
| `google-ai-edge/ml-drift` | pull requests | Google CLA (released 2026-10-08) |

| Draft | Kind | Track | Our commits |
| --- | --- | --- | --- |
| [litert-lm-swift-stream-stall.md](litert-lm-swift-stream-stall.md) | LiteRT-LM issue | — | `agello` 85b1a776 |
| [litert-lm-swift-tool-call-errors.md](litert-lm-swift-tool-call-errors.md) | LiteRT-LM issue | — | `agello` a3c0d0ef |
| [litert-lm-session-snapshots.md](litert-lm-session-snapshots.md) | LiteRT-LM issue | T3 | `agello-next` aec81cb7, ed998d80, 585203d3 |
| [litert-lm-decode-host-costs.md](litert-lm-decode-host-costs.md) | LiteRT-LM issue | T2, P6 | `agello-next` ef197a04, e43f9a8d, 5e946049, fdd5bac7 |
| [litert-bounded-attention.md](litert-bounded-attention.md) | LiteRT PR + ML Drift PR | T4 | LiteRT 31386317 and earlier; ML Drift 6ae7c68, 2184a7f |
| [litert-weight-cache-key.md](litert-weight-cache-key.md) | LiteRT PR | T11 | LiteRT 97e06d36 |
| [litert-metal-direct-copies.md](litert-metal-direct-copies.md) | LiteRT PR | T5 | LiteRT `agello-next` (T5 step 1) |

Every draft states its measurements on the M6 Mac. An iPhone run should back them before
anything is posted.
