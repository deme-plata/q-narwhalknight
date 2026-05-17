// Codec negotiation and bitrate estimation — pure computation, no system calls.
// Extracted from nova-chat src/media/codec.rs (computation logic only).
// encode/decode methods excluded — browser handles actual media encoding via WebRTC API.

use super::call::Resolution;

#[derive(Debug, Clone, PartialEq)]
pub enum AudioCodec {
    Opus,
    Pcm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VideoCodec {
    H264,
    Vp8,
    Vp9,
}

pub struct CodecManager;

impl CodecManager {
    pub fn new() -> Self {
        Self
    }

    /// Select optimal audio codec for the given bandwidth.
    pub fn optimal_audio_codec(&self, bandwidth_kbps: u32) -> AudioCodec {
        if bandwidth_kbps >= 128 {
            AudioCodec::Opus
        } else {
            AudioCodec::Pcm
        }
    }

    /// Select optimal video codec for the given bandwidth and resolution.
    pub fn optimal_video_codec(&self, bandwidth_kbps: u32, resolution: &Resolution) -> VideoCodec {
        if bandwidth_kbps >= 2000 && resolution.width >= 1920 {
            VideoCodec::Vp9
        } else if bandwidth_kbps >= 1000 {
            VideoCodec::H264
        } else {
            VideoCodec::Vp8
        }
    }

    /// Estimate required bitrate in kbps for the given codec, resolution, and fps.
    pub fn estimate_bitrate_kbps(&self, codec: &VideoCodec, resolution: &Resolution, fps: u32) -> u32 {
        let base = match codec {
            VideoCodec::H264 => {
                if resolution.width >= 1920 { 4000 }
                else if resolution.width >= 1280 { 2000 }
                else { 1000 }
            }
            VideoCodec::Vp8 => {
                if resolution.width >= 1920 { 5000 }
                else if resolution.width >= 1280 { 2500 }
                else { 1200 }
            }
            VideoCodec::Vp9 => {
                if resolution.width >= 1920 { 3000 }
                else if resolution.width >= 1280 { 1500 }
                else { 800 }
            }
        };
        (base * fps / 30).max(500)
    }
}

impl Default for CodecManager {
    fn default() -> Self {
        Self::new()
    }
}
