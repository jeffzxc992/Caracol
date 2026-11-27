# Caracol

Caracol is a macOS application for real-time frame interpolation and game capture. It captures the screen using ScreenCaptureKit, interpolates frames with a CoreML model (IFRNetLite), and renders the result with Metal/MetalFX to deliver smoother gameplay visuals.

## Features

- Screen capture via ScreenCaptureKit
- Frame interpolation powered by CoreML (IFRNetLite)
- Metal / MetalFX rendering pipeline
- Configurable capture and display settings
- Powered by Apple Neural Engine

## Requirements

- macOS 26.0 or later
- Apple Silicon M1 or later

## Project structure

```
Caracol/
├── Capture/          # Screen capture engine
├── Models/           # Data models and settings
├── Resources/        # Info.plist, entitlements, ML models
├── Support/          # Math, logging, geometry, metrics
└── Telemetry/        # Metrics store
project.yml           # XcodeGen project definition
```
