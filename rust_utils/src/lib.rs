#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
// Many false positives with pyo3 it seems &str, and &PyAny get flagged
#![allow(clippy::borrow_deref_ref)]

extern crate tokenizers as tk;

use tk::pre_tokenizers::split::SplitPattern;
use tk::tokenizer::{PreTokenizedString, PreTokenizer};
use tk::{OffsetReferential, OffsetType, SplitDelimiterBehavior};

use rand_distr::{Distribution, Normal};


use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyclass(name = "TokenizerSampler")]
#[derive(Clone)]
pub struct PyTokenizerSampler {
    sampler: Arc<RwLock<TokenizerSampler>>
}

const SPLIT_REGEX: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[pymethods]
impl PyTokenizerSampler {
    #[new]
    fn new() -> Self {
        PyTokenizerSampler {
            sampler: Arc::new(RwLock::new(TokenizerSampler {
                seed_cache: VecDeque::new(),
            }))
        }
    }

    #[pyo3(text_signature = "(self, map, seed_size, max_length, stride, noise_std, pop_prev, push_current)")]
    fn sample_tokenizer(
        self_: PyRef<Self>,
        map: HashMap<String, u32>,
        seed_size: usize,
        max_length: usize,
        stride: Option<usize>,
        noise_std: Option<f64>,
        pop_prev: Option<bool>,
        push_current: Option<bool>,
    ) -> PyResult<Vec<(String, f64)>> {
        let mut model: std::sync::RwLockWriteGuard<'_, TokenizerSampler> = self_.sampler.write().unwrap();

        Ok(model.sample_tokenizer(
            map,
            seed_size,
            max_length,
            stride.unwrap_or(1),
            noise_std.unwrap_or(0.0),
            pop_prev.unwrap_or(true),
            push_current.unwrap_or(true),
        ))
    }
}

pub struct TokenizerSampler {
    seed_cache: VecDeque<HashMap<String, u32>>,
}

impl TokenizerSampler {
    pub fn sample_tokenizer(
        &mut self,
        map: HashMap<String, u32>,
        seed_size: usize,
        max_length: usize,
        stride: usize,
        noise_std: f64,
        pop_prev: bool,
        push_current: bool,
    ) -> Vec<(String, f64)> {
        let mut seed_sentencepieces: Vec<(String, f64)> = vec![];

        let pre_tokenizer = tk::pre_tokenizers::sequence::Sequence::new(
            vec![
                tk::pre_tokenizers::split::Split::new(
                    SplitPattern::Regex(SPLIT_REGEX.into()),
                    SplitDelimiterBehavior::Removed,
                    true,
                ).unwrap().into(),
                tk::pre_tokenizers::byte_level::ByteLevel::new(false, true, false).into(),
            ]
        );

        let data: Vec<(PreTokenizedString, Vec<usize>, u32)> = map
            .into_iter()
            .map(|(sentence, n)| {
                let mut sentence = sentence;
                sentence.insert_str(0, " "); // prefix space

                let cumulative_byte_length: Vec<_> = sentence
                    .chars()
                    .scan(0, |acc, x| {
                        *acc += x.len_utf8();
                        Some(*acc)
                    })
                    .collect();

                let mut pretokenized: PreTokenizedString = sentence.into();
                pre_tokenizer.pre_tokenize(&mut pretokenized).unwrap();
                (pretokenized, cumulative_byte_length, n)
            })
            .collect();

        let mut current_substr_index: HashMap<&str, u32> = HashMap::new();

        for (pretokenized, cumulative_byte_length, n) in data.iter() {
            for (i, (pretoken, offsets, _)) in pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Char)
                .iter()
                .enumerate()
            {
                let mut byte_offsets: Vec<_> = cumulative_byte_length[offsets.0..offsets.1]
                    .iter()
                    .map(|x| x - cumulative_byte_length[offsets.0])
                    .collect();
                if i == 0 {
                    byte_offsets.insert(0, 0);
                }

                let pretoken_char_offsets: Vec<_> = pretoken.char_indices().collect();

                for i in byte_offsets.iter().step_by(stride) {
                    for k in 1..max_length {
                        if i + k > pretoken_char_offsets.len() {
                            break;
                        }
                        let start = pretoken_char_offsets[*i].0;
                        let end = if i + k == pretoken_char_offsets.len() {
                            pretoken.len()
                        } else {
                            pretoken_char_offsets[*i + k].0
                        };
                        let token = &pretoken[start..end];

                        if token.is_empty() {
                            continue;
                        }

                        let freq = *n;
                        let score = freq * token.len() as u32;

                        current_substr_index
                            .entry(token)
                            .and_modify(|e| *e += score)
                            .or_insert(score);
                    }
                }
            }
        }

        let maybe_prev = if pop_prev {
            self.seed_cache.pop_back()
        } else {
            None
        };

        self.seed_cache.push_front(
            current_substr_index
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        );

        if pop_prev {
            let substr_index: HashMap<&str, u32> =
                self.seed_cache.iter().map(|x| x.iter()).flatten().fold(
                    HashMap::new(),
                    |mut acc, (k, v)| {
                        acc.entry(k.as_str()).and_modify(|e| *e += *v).or_insert(*v);
                        acc
                    },
                );

            let score_sum = substr_index.iter().map(|x| x.1).sum::<u32>() as f64;
            let min_score = substr_index.iter().fold(u32::MAX, |a, b| a.min(*b.1)) as f64;
            let min_log_prob = (min_score / score_sum).ln();

            // Fill seed_sentencepieces
            for character in tk::pre_tokenizers::byte_level::ByteLevel::alphabet() {
                let string = character.to_string();
                seed_sentencepieces.push((string, min_log_prob));
            }

            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, noise_std).unwrap();
            let mut substr_index = substr_index
                .into_iter()
                .map(|(x, v)| {
                    let noised = v as f64 / score_sum as f64 + normal.sample(&mut rng);

                    if noised > 0.0 {
                        (x, noised.ln())
                    } else {
                        (x, -100000.0)
                    }
                })
                .collect::<Vec<_>>();

            // sort by decreasing score
            substr_index.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let extra_whitespace_chars = ['Ġ', 'Ċ', 'ĉ'];

            for c1 in extra_whitespace_chars.iter() {
                // length 1 already added through ByteLevel::alphabet()
                for i in 1..max_length {
                    for c2 in extra_whitespace_chars.iter() {
                        let string = c2.to_string() + c1.to_string().repeat(i).as_str();

                        seed_sentencepieces.push((string, 0.0));
                    }
                }
            }

            for (string, score) in substr_index {
                if string.chars().count() == 1
                    || string
                        .chars()
                        .map(|c| extra_whitespace_chars.contains(&c) as usize)
                        .sum::<usize>()
                        >= 2
                {
                    // already added
                    continue;
                }
                seed_sentencepieces.push((string.into(), score));
                if seed_sentencepieces.len() >= seed_size {
                    break;
                }
            }
        }
        if !push_current {
            self.seed_cache.pop_front();
            if let Some(prev) = maybe_prev {
                self.seed_cache.push_back(prev);
            }
        }

        seed_sentencepieces
    }
}

#[pymodule]
pub fn rust_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizerSampler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
