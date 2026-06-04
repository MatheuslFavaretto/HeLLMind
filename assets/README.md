# assets

Drop a **`doomguy.png`** here (the real DOOM HUD marine face) and `doom-cli shell` will render
the **actual game image** inline as its backdrop — on terminals that support inline images
(**iTerm2** on macOS). Any other terminal falls back to the built-in ASCII Doomguy.

```
assets/doomguy.png   ← put it here
```

Where to get one: screenshot the DOOM status-bar face, or grab the `STFST*` sprite from a
DOOM sprite sheet. A square-ish PNG (e.g. 256×256) on a dark/transparent background looks best.

Then just:

```bash
doom-cli shell
```
