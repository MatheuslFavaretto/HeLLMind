---
type: control
stop_training: false
novelty_threshold: 0.15
write_every_steps: 50000
---

# Training control panel

Edit the **frontmatter** values above while training runs — it re-reads this
file every few thousand steps and adapts without restarting.

- `stop_training: true` ends training cleanly (saves the model).
- `novelty_threshold` controls how different something must be to become a note.
- `write_every_steps` controls the collection cadence.
