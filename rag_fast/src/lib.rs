use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

fn tokenize_impl(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            for lc in ch.to_lowercase() {
                buf.push(lc);
            }
        } else if !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
        }
    }

    if !buf.is_empty() {
        out.push(buf);
    }

    out
}

#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    tokenize_impl(text)
}

#[pyclass]
struct BM25Index {
    k1: f32,
    b: f32,
    avgdl: f32,
    doc_len: Vec<usize>,
    // Per-doc term frequency maps
    tf: Vec<HashMap<String, u32>>,
    // Document frequency per term
    df: HashMap<String, u32>,
    n_docs: usize,
}

#[pymethods]
impl BM25Index {
    #[new]
    fn new(k1: Option<f32>, b: Option<f32>) -> Self {
        BM25Index {
            k1: k1.unwrap_or(1.5),
            b: b.unwrap_or(0.75),
            avgdl: 0.0,
            doc_len: Vec::new(),
            tf: Vec::new(),
            df: HashMap::new(),
            n_docs: 0,
        }
    }

    #[staticmethod]
    fn from_tokenized(docs: Vec<Vec<String>>, k1: Option<f32>, b: Option<f32>) -> Self {
        let mut idx = BM25Index::new(k1, b);
        for doc in docs {
            idx.add_doc(doc);
        }
        idx
    }

    fn add_doc(&mut self, doc_tokens: Vec<String>) {
        let mut map: HashMap<String, u32> = HashMap::new();
        for t in doc_tokens.iter() {
            *map.entry(t.clone()).or_insert(0) += 1;
        }

        let dl = doc_tokens.len();
        self.doc_len.push(dl);
        self.tf.push(map);
        self.n_docs += 1;

        let mut seen: HashSet<&String> = HashSet::new();
        for t in doc_tokens.iter() {
            if seen.insert(t) {
                *self.df.entry(t.clone()).or_insert(0) += 1;
            }
        }

        let sum_dl: usize = self.doc_len.iter().sum();
        self.avgdl = if self.n_docs == 0 {
            0.0
        } else {
            (sum_dl as f32) / (self.n_docs as f32)
        };
    }

    fn topk(&self, query_tokens: Vec<String>, k: usize) -> Vec<usize> {
        if self.n_docs == 0 || k == 0 {
            return Vec::new();
        }

        let mut q_terms: HashMap<String, u32> = HashMap::new();
        for t in query_tokens {
            *q_terms.entry(t).or_insert(0) += 1;
        }

        let n = self.n_docs as f32;
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.n_docs);

        for (doc_id, tf_map) in self.tf.iter().enumerate() {
            let dl = self.doc_len[doc_id] as f32;
            let denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl.max(1e-6)));

            let mut score: f32 = 0.0;
            for (term, _qf) in q_terms.iter() {
                let df = match self.df.get(term) {
                    Some(v) => *v as f32,
                    None => continue,
                };
                let tf = match tf_map.get(term) {
                    Some(v) => *v as f32,
                    None => continue,
                };

                // BM25 idf with +1 inside log to avoid negative infinities for very frequent terms.
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                let numer = tf * (self.k1 + 1.0);
                score += idf * (numer / (tf + denom_norm));
            }
            scored.push((doc_id, score));
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
        });

        scored.into_iter().take(k.min(self.n_docs)).map(|x| x.0).collect()
    }
}

#[pymodule]
fn rag_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_class::<BM25Index>()?;
    Ok(())
}

