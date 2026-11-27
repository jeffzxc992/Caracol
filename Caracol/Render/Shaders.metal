#include <metal_stdlib>
using namespace metal;

struct RenderUniforms {
    float4 p0; // x: t, y: flow scale, z: reserved, w: sharpness
    float4 p1; // x: src width, y: src height, z: dst width, w: dst height
    float4 p2; // x: source aspect, y: letterbox flag, z: mode, w: reserved
    float4 p3; // x: interpolation flag, y: occlusion threshold, z: block size, w: search radius
};

constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);

inline float lumaOf(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// MARK: - Luma downsample (full resolution BGRA -> 1/4 R16F)

kernel void luma_downsample(texture2d<float> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            constant RenderUniforms &u [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint dw = dst.get_width();
    uint dh = dst.get_height();
    if (gid.x >= dw || gid.y >= dh) { return; }

    uint sw = src.get_width();
    uint sh = src.get_height();
    uint factor = max(1u, uint(u.p0.z));
    uint2 base = gid * factor;
    float sum = 0.0;
    float count = 0.0;
    for (uint y = 0; y < factor; ++y) {
        for (uint x = 0; x < factor; ++x) {
            uint2 p = min(base + uint2(x, y), uint2(sw - 1, sh - 1));
            sum += lumaOf(src.read(p).rgb);
            count += 1.0;
        }
    }
    dst.write(float(sum / max(count, 1.0)), gid);
}

// MARK: - Block matching (1/4 luma -> coarse flow, normalized UV)

kernel void block_match(texture2d<float> prevLuma [[texture(0)]],
                        texture2d<float> curLuma [[texture(1)]],
                        texture2d<float, access::write> coarse [[texture(2)]],
                        constant RenderUniforms &u [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint bw = coarse.get_width();
    uint bh = coarse.get_height();
    if (gid.x >= bw || gid.y >= bh) { return; }

    int block = max(4, int(u.p3.z));
    int radius = max(2, int(u.p3.w));
    int2 size = int2(curLuma.get_width(), curLuma.get_height());
    int2 center = int2(gid) * block + block / 2;

    float best = 1e20;
    int2 bestOffset = int2(0);

    // A 3x3 luma window per candidate: much cheaper than a full block SAD and
    // robust enough for a first motion estimate at 1/4 resolution.
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float sad = 0.0;
            for (int sy = -1; sy <= 1; ++sy) {
                for (int sx = -1; sx <= 1; ++sx) {
                    int2 pc = clamp(center + int2(sx, sy), int2(0), size - 1);
                    int2 pp = clamp(center + int2(dx, dy) + int2(sx, sy), int2(0), size - 1);
                    sad += fabs(curLuma.read(uint2(pc)).r - prevLuma.read(uint2(pp)).r);
                }
            }
            if (sad < best) {
                best = sad;
                bestOffset = int2(dx, dy);
            }
        }
    }

    // Normalized displacement: cur pixel -> prev pixel. Constant velocity
    // extrapolation samples cur at uv + flow * t.
    float2 flow = float2(bestOffset) * float(block) / float2(size);
    float confidence = saturate(1.0 - best / 9.0 * 3.0);
    coarse.write(float4(flow, confidence, best / 9.0), gid);
}

// MARK: - Flow refinement (1/4 RGBA16F, normalized UV, confidence in .z)

kernel void flow_refine(texture2d<float> prevLuma [[texture(0)]],
                        texture2d<float> curLuma [[texture(1)]],
                        texture2d<float> flowIn [[texture(2)]],
                        texture2d<float, access::write> flowOut [[texture(3)]],
                        constant RenderUniforms &u [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint2 size = uint2(flowOut.get_width(), flowOut.get_height());
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    int2 isize = int2(size);
    int2 center = int2(gid);
    float2 uv = (float2(gid) + 0.5) / float2(size);

    float4 initial = flowIn.sample(linearSampler, uv);
    int2 initialPixels = int2(round(initial.xy * float2(size)));
    int radius = 2;
    float best = 1e20;
    int2 bestOffset = initialPixels;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int2 d = initialPixels + int2(dx, dy);
            float sad = 0.0;
            for (int sy = -1; sy <= 1; ++sy) {
                for (int sx = -1; sx <= 1; ++sx) {
                    int2 pc = clamp(center + int2(sx, sy), int2(0), isize - 1);
                    int2 pp = clamp(center + d + int2(sx, sy), int2(0), isize - 1);
                    sad += fabs(curLuma.read(uint2(pc)).r - prevLuma.read(uint2(pp)).r);
                }
            }
            if (sad < best) {
                best = sad;
                bestOffset = d;
            }
        }
    }

    float2 flow = float2(bestOffset) / float2(size);
    float error = best / 9.0;
    float confidence = saturate(1.0 - error * 4.0) * initial.z;
    flowOut.write(float4(flow, confidence, error), gid);
}

// MARK: - Fullscreen triangle

struct VSOut {
    float4 position [[position]];
    float2 uv;
};

vertex VSOut fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(uv.x, 1.0 - uv.y);
    return out;
}

// MARK: - Warp / extrapolate / interpolate

inline float4 warpColor(float2 uv,
                        texture2d<float> cur,
                        texture2d<float> prev,
                        texture2d<float> flow,
                        constant RenderUniforms &u) {
    float t = saturate(u.p0.x);
    float3 base = cur.sample(linearSampler, uv).rgb;

    if (t <= 0.0001) {
        return float4(base, 1.0);
    }

    float4 f = flow.sample(linearSampler, uv);
    float2 motion = f.xy;
    float confidence = saturate(f.z);

    // Optional interpolation: sample both real frames at the intermediate time.
    if (u.p3.x > 0.5) {
        float3 a = prev.sample(linearSampler, uv + motion * (1.0 - t)).rgb;
        float3 b = cur.sample(linearSampler, uv - motion * t).rgb;
        return float4(mix(a, b, t), 1.0);
    }

    // Extrapolation: continue constant velocity past the newest real frame.
    // flow maps cur -> prev, so the future sample is cur at uv - flow * t.
    float2 sampleUV = uv - motion * t;
    float3 warped = cur.sample(linearSampler, sampleUV).rgb;
    bool inside = sampleUV.x >= 0.0 && sampleUV.x <= 1.0 &&
                  sampleUV.y >= 0.0 && sampleUV.y <= 1.0;
    float colorDiff = distance(warped, base);
    float w = confidence * (inside ? 1.0 : 0.0);
    w *= saturate(1.0 - colorDiff * 1.6);
    w *= saturate(1.0 - length(motion * t) * 1.5);
    return float4(mix(base, warped, w), 1.0);
}

kernel void warp_compute(texture2d<float> cur [[texture(0)]],
                         texture2d<float> prev [[texture(1)]],
                         texture2d<float> flow [[texture(2)]],
                         texture2d<float, access::write> outTex [[texture(3)]],
                         constant RenderUniforms &u [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint w = outTex.get_width();
    uint h = outTex.get_height();
    if (gid.x >= w || gid.y >= h) { return; }
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    outTex.write(warpColor(uv, cur, prev, flow, u), gid);
}

fragment float4 warp_fragment(VSOut in [[stage_in]],
                              texture2d<float> cur [[texture(0)]],
                              texture2d<float> prev [[texture(1)]],
                              texture2d<float> flow [[texture(2)]],
                              constant RenderUniforms &u [[buffer(0)]]) {
    return warpColor(in.uv, cur, prev, flow, u);
}

// MARK: - CAS-like sharpening + aspect-fit presentation

fragment float4 cas_fragment(VSOut in [[stage_in]],
                             texture2d<float> src [[texture(0)]],
                             constant RenderUniforms &u [[buffer(0)]]) {
    float outAspect = u.p1.z / max(u.p1.w, 1.0);
    float srcAspect = u.p2.x;
    float2 uv = in.uv;

    if (u.p2.y > 0.5 && fabs(outAspect - srcAspect) > 0.0005) {
        if (outAspect > srcAspect) {
            uv.x = (uv.x - 0.5) * (srcAspect / outAspect) + 0.5;
        } else {
            uv.y = (uv.y - 0.5) * (outAspect / srcAspect) + 0.5;
        }
    }
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float3 c = src.sample(linearSampler, uv).rgb;
    float3 blur = float3(0.0);
    blur += src.sample(linearSampler, uv + float2( texel.x, 0.0)).rgb;
    blur += src.sample(linearSampler, uv + float2(-texel.x, 0.0)).rgb;
    blur += src.sample(linearSampler, uv + float2(0.0,  texel.y)).rgb;
    blur += src.sample(linearSampler, uv + float2(0.0, -texel.y)).rgb;
    blur += src.sample(linearSampler, uv + texel).rgb;
    blur += src.sample(linearSampler, uv - texel).rgb;
    blur += src.sample(linearSampler, uv + float2(texel.x, -texel.y)).rgb;
    blur += src.sample(linearSampler, uv + float2(-texel.x, texel.y)).rgb;
    blur *= 0.125;

    float sharpness = saturate(u.p0.w);
    float3 outc = saturate(c + (c - blur) * (0.35 + sharpness * 1.8));
    return float4(outc, 1.0);
}
