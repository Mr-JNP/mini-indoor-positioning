# Disclaimer

This code is borrowed by: [Social Distancing AI](https://github.com/deepak112/Social-Distancing-AI)

To run:

```
python -m app.main --video_path [INPUT VIDEO PATH] --bounding_boxes [BOUNDING BOXES PATH] --output_vid [OUTPUT VIDEO PATH]
```

Use FFmpeg to covert from .avi to .mp4:

```
ffmpeg -i output/example.avi -c:v copy -c:a copy -y output/example.mp4
```
