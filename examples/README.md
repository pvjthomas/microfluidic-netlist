# Example DXF Files

This directory contains example DXF files and utilities for generating test cases.

## Generate Y-Channel Test File

Run the generator script to create a test Y-channel DXF:

```bash
cd examples
python generate_y_channel.py
```

This creates `y_channel_100um_4mmlegs.dxf` - a Y-shaped microfluidic channel with:
- 100µm channel width
- 4mm leg length
- 45° angle between arms

The generated file can be used to test the DXF loader and graph extraction pipeline.

