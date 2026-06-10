// GGUF Tokenizer - Adapted from mistral.rs
// Original: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/gguf/gguf_tokenizer.rs
//
// This module provides GGUF tokenizer conversion functionality for q-ai-inference.
// Converts GGUF metadata-embedded tokenizers to HuggingFace tokenizers format.

use std::collections::HashMap;
use ahash::AHashMap;
use anyhow::Result;
use candle_core::quantized::gguf_file::Value;
use itertools::Itertools;
use tokenizers::pre_tokenizers::{
    sequence::Sequence,
    split::{Split, SplitPattern},
    PreTokenizerWrapper,
};
use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;
use tokenizers::{
    decoders::{
        byte_fallback::ByteFallback, byte_level::ByteLevel, fuse::Fuse, strip::Strip,
    },
    models::{bpe::BpeBuilder, unigram::Unigram},
    normalizers::{Prepend, Replace},
    processors, AddedToken, DecoderWrapper, ModelWrapper, NormalizerWrapper, Tokenizer,
};
use tracing::info;

/// Result of GGUF tokenizer conversion containing the tokenizer and special tokens
pub struct GgufTokenizerConversion {
    pub tokenizer: Tokenizer,
    pub bos: Option<String>,
    pub eos: Option<String>,
    pub unk: Option<String>,
}

/// Internal structure holding GGUF tokenizer properties
struct PropsGGUF {
    model: String,
    tokens: Vec<String>,
    added_tokens: Option<Vec<String>>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
    unk: Option<u32>,
    bos: Option<u32>,
    eos: u32,
}

/// Helper struct for accessing GGUF metadata with a path prefix
pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, Value>,
}

impl ContentMetadata<'_> {
    /// Retrieve a value from metadata with type conversion
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T> {
        let prop_key = format!("{}.{}", self.path_prefix, field_name);
        let value = self.metadata.get(&prop_key).cloned();

        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{}` `{}`", prop_key, e))
    }

    /// Check that all required keys are present
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_present = true;

        for field_name in fields {
            let prop_key = format!("{}.{}", self.path_prefix, field_name);
            if !self.metadata.contains_key(&prop_key) {
                all_props_present = false;
                tracing::warn!("Expected GGUF metadata to have key: `{}`", prop_key);
            }
        }

        anyhow::ensure!(all_props_present, "Tokenizer is missing required props");
        Ok(())
    }
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> Result<Self> {
        let required = ["model", "tokens", "eos_token_id"];
        c.has_required_keys(&required)?;

        Ok(Self {
            model: c.get_value("model")?,
            tokens: c.get_value("tokens")?,
            added_tokens: c.get_value("added_tokens").ok(),
            scores: c.get_value("scores").ok(),
            merges: c.get_value("merges").ok(),
            unk: c.get_value("unknown_token_id").ok(),
            eos: c.get_value("eos_token_id")?,
            bos: c.get_value("bos_token_id").ok(),
        })
    }
}

/// Convert GGUF metadata to HuggingFace tokenizer
pub fn convert_gguf_to_hf_tokenizer(
    metadata: &HashMap<String, Value>
) -> Result<GgufTokenizerConversion> {
    let content_metadata = ContentMetadata {
        path_prefix: "tokenizer.ggml",
        metadata,
    };

    // Extract token types if present
    let mut token_types = Vec::<i32>::new();
    if metadata.contains_key("tokenizer.ggml.token_type") {
        if let Some(Value::Array(vtypes)) = metadata.get("tokenizer.ggml.token_type") {
            token_types = vtypes.iter()
                .filter_map(|v| v.to_i32().ok())
                .collect();
        }
    }

    let props = PropsGGUF::try_from(content_metadata)?;

    let (mut tokenizer, kind) = match props.model.as_str() {
        "llama" | "replit" => unigram_tokenizer(&props)?,
        "gpt2" => bpe_tokenizer(&props)?,
        other => {
            anyhow::bail!("Tokenizer model `{}` not supported.", other);
        }
    };

    // Add special tokens (token types other than 1 are treated as special)
    let mut num_special_tokens = 0;
    if token_types.len() == props.tokens.len() {
        for i in 0..props.tokens.len() {
            if token_types[i] != 1 {
                let tk = props.tokens[i].clone();
                tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
                num_special_tokens += 1;
            }
        }
    }

    info!(
        "GGUF tokenizer model: `{}`, kind: `{:?}`, vocab size: {}, special tokens: {}, added tokens: {}, merges: {}, scores: {}",
        props.model,
        kind,
        tokenizer.get_vocab_size(true),
        num_special_tokens,
        props.added_tokens.as_ref().map(|x| x.len()).unwrap_or(0),
        props.merges.as_ref().map(|x| x.len()).unwrap_or(0),
        props.scores.as_ref().map(|x| x.len()).unwrap_or(0),
    );

    let unk = props.unk.map(|u| props.tokens[u as usize].clone());
    let bos = props.bos.map(|b| props.tokens[b as usize].clone());
    let eos = Some(props.tokens[props.eos as usize].clone());

    Ok(GgufTokenizerConversion {
        tokenizer,
        bos,
        eos,
        unk,
    })
}

#[derive(Debug)]
enum TokenizerKind {
    Unigram,
    Bpe,
}

/// Create a Unigram (SentencePiece) tokenizer
fn unigram_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    let PropsGGUF { unk, eos, bos, .. } = *p;
    let unk = unk.unwrap_or(0); // Unigram default UNK is 0

    // Create vocab with scores
    let model = {
        let Some(scores) = p.scores.as_ref() else {
            anyhow::bail!(
                "`llama` unigram tokenizer is missing required metadata `tokenizer.ggml.scores`"
            );
        };

        let vocab: Vec<(String, f64)> = p.tokens.iter()
            .cloned()
            .zip(scores.iter().map(|&f| f as f64))
            .collect();

        Unigram::from(vocab, Some(unk as usize), true)
            .map_err(anyhow::Error::msg)?
    };

    let decoder = Decoder::Sequence(vec![
        Decoder::Replace("▁", " "),
        Decoder::ByteFallback,
        Decoder::Fuse,
        Decoder::Strip(' ', 1, 0),
    ]);

    let normalizer = Normalizer::Sequence(vec![
        Normalizer::Prepend("▁"),
        Normalizer::Replace(" ", "▁"),
    ]);

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::Unigram(model),
        Some(decoder),
        Some(normalizer),
    )?;

    // Add special tokens
    for v in [bos, Some(eos), Some(unk)].iter().flatten() {
        let tk = p.tokens[*v as usize].clone();
        tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
    }

    Ok((tokenizer, TokenizerKind::Unigram))
}

/// Create a BPE tokenizer
fn bpe_tokenizer(p: &PropsGGUF) -> Result<(Tokenizer, TokenizerKind)> {
    // BPE merges have each string item as a space-delimited pair
    let merges = p.merges
        .as_ref()
        .ok_or_else(|| anyhow::Error::msg("BPE tokenizer must include merges"))?
        .iter()
        .map(|merge| {
            let (a, b) = merge.splitn(2, ' ')
                .collect_tuple()
                .expect("Failed to parse BPE merge");
            (a.to_string(), b.to_string())
        })
        .collect::<Vec<_>>();

    let mut vocab = AHashMap::new();
    for (i, token) in p.tokens.iter().enumerate() {
        vocab.insert(token.clone(), i as u32);
    }

    let PropsGGUF { bos, eos, unk, .. } = *p;

    let mut bpe = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = unk {
        bpe = bpe.unk_token(p.tokens[unk as usize].to_string());
    }

    let bpe = bpe.build().map_err(anyhow::Error::msg)?;

    let mut tokenizer = TokenizerX::new(
        ModelWrapper::BPE(bpe),
        Some(Decoder::ByteLevel(true, true, true)),
        None,
    )?;

    let split = Split::new(
        SplitPattern::Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    ).unwrap();

    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]);

    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_post_processor(Some(processors::byte_level::ByteLevel::new(
        false, false, false,
    )));

    // Add special tokens
    for v in [bos, Some(eos), unk].iter().flatten() {
        let tk = p.tokens[*v as usize].clone();
        tokenizer.add_special_tokens(&[AddedToken::from(tk, true)]);
    }

    Ok((tokenizer, TokenizerKind::Bpe))
}

// ============================================================================
// Helper Types for Tokenizer Construction
// ============================================================================

/// Workaround for better builder API (upstream TokenizerBuilder is difficult to use)
struct TokenizerX;

impl TokenizerX {
    fn new<'a>(
        model: ModelWrapper,
        decoder: Option<Decoder<'a>>,
        normalizer: Option<Normalizer<'a>>,
    ) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::new(model);

        if let Some(decoder) = decoder {
            let d = DecoderWrapper::try_from(decoder)?;
            tokenizer.with_decoder(Some(d));
        }
        if let Some(normalizer) = normalizer {
            let n = NormalizerWrapper::try_from(normalizer)?;
            tokenizer.with_normalizer(Some(n));
        }

        Ok(tokenizer)
    }
}

/// Convenient alternative to upstream DecoderWrapper enum
enum Decoder<'a> {
    ByteFallback,
    Fuse,
    Replace(&'a str, &'a str),
    Strip(char, usize, usize),
    Sequence(Vec<Self>),
    ByteLevel(bool, bool, bool),
}

impl TryFrom<Decoder<'_>> for DecoderWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Decoder) -> Result<Self> {
        let value = match variant {
            Decoder::ByteFallback => ByteFallback::default().into(),
            Decoder::Fuse => Fuse::default().into(),
            Decoder::Replace(pattern, content) => {
                Replace::new(pattern, content)
                    .map_err(anyhow::Error::msg)?
                    .into()
            }
            Decoder::Strip(content, start, stop) => Strip::new(content, start, stop).into(),
            Decoder::Sequence(decoders) => {
                let seq = decoders
                    .into_iter()
                    .map(DecoderWrapper::try_from)
                    .collect::<Result<Vec<_>>>()?;
                tokenizers::decoders::sequence::Sequence::new(seq).into()
            }
            Decoder::ByteLevel(add_prefix_space, trim_offsets, use_regex) => {
                ByteLevel::new(add_prefix_space, trim_offsets, use_regex).into()
            }
        };
        Ok(value)
    }
}

/// Convenient alternative to upstream NormalizerWrapper enum
enum Normalizer<'a> {
    Prepend(&'a str),
    Replace(&'a str, &'a str),
    Sequence(Vec<Self>),
}

impl TryFrom<Normalizer<'_>> for NormalizerWrapper {
    type Error = anyhow::Error;

    fn try_from(variant: Normalizer) -> Result<Self> {
        let value = match variant {
            Normalizer::Prepend(prepend) => Prepend::new(prepend.to_owned()).into(),
            Normalizer::Replace(pattern, content) => {
                Replace::new(pattern, content)
                    .map_err(anyhow::Error::msg)?
                    .into()
            }
            Normalizer::Sequence(normalizers) => {
                let seq = normalizers
                    .into_iter()
                    .map(NormalizerWrapper::try_from)
                    .collect::<Result<Vec<_>>>()?;
                tokenizers::normalizers::Sequence::new(seq).into()
            }
        };
        Ok(value)
    }
}

// ============================================================================
// Type Conversion Traits for GGUF Values
// ============================================================================

/// Trait for converting from GGUF Value types
pub trait TryFromValue {
    fn try_from_value(value: Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

/// Trait for converting GGUF Values into other types
pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Expected value but got None"),
        }
    }
}

// Implement TryFromValue for common types
macro_rules! impl_try_from_value {
    ($typ:ty, $method:ident) => {
        impl TryFromValue for $typ {
            fn try_from_value(value: Value) -> Result<Self, candle_core::Error> {
                value.$method()
                    .or_else(|_| candle_core::bail!(concat!("value is not a `", stringify!($typ), "`")))
            }
        }
    };
}

impl_try_from_value!(bool, to_bool);
impl_try_from_value!(f32, to_f32);
impl_try_from_value!(f64, to_f64);
impl_try_from_value!(i8, to_i8);
impl_try_from_value!(i16, to_i16);
impl_try_from_value!(i32, to_i32);
impl_try_from_value!(i64, to_i64);
impl_try_from_value!(u8, to_u8);
impl_try_from_value!(u16, to_u16);
impl_try_from_value!(u32, to_u32);
impl_try_from_value!(u64, to_u64);

impl TryFromValue for String {
    fn try_from_value(value: Value) -> Result<Self, candle_core::Error> {
        value.to_string()
            .cloned()
            .or_else(|_| candle_core::bail!("value is not a `String`"))
    }
}

impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(T::try_from_value)
            .collect()
    }
}
