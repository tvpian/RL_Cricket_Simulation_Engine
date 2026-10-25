# Original project archive

`kivy/` contains the original scripts, images, video and saved checkpoint, moved
without altering their contents. The original repository description is preserved
in `kivy/README.original.md`.

To try the historical visual demo on a desktop with a display:

```bash
pip install -e '.[gui]'
cd legacy/kivy
python main.py
```

The original README describes a trained RL bowler and lists a trained batsman as
future work. The saved `last_brain.pth` contains neural weights, but this checkout
does not include a complete, reproducible training pipeline connected to it. The
current course agents are separate implementations; we do not claim to have
recovered or continued the original model's training.
