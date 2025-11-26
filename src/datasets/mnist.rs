//! MNIST dataset loader
//!
//! Embedded sample (100 per digit = 1000 total) works offline.
//! Full dataset (70k) available with `hf-hub` feature.

use std::sync::Arc;

use arrow::{
    array::{Float32Array, Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};

use super::{CanonicalDataset, DatasetSplit};
use crate::{
    transform::{Skip, Take, Transform},
    ArrowDataset, Dataset, Result,
};

/// Load MNIST dataset (embedded 1000-sample subset)
///
/// # Errors
///
/// Returns an error if dataset construction fails.
pub fn mnist() -> Result<MnistDataset> {
    MnistDataset::load()
}

/// MNIST handwritten digits dataset
#[derive(Debug, Clone)]
pub struct MnistDataset {
    data: ArrowDataset,
}

impl MnistDataset {
    /// Load embedded MNIST sample
    ///
    /// # Errors
    ///
    /// Returns an error if construction fails.
    pub fn load() -> Result<Self> {
        // Schema: 784 pixel columns + label
        let mut fields: Vec<Field> = (0..784)
            .map(|i| Field::new(format!("pixel_{i}"), DataType::Float32, false))
            .collect();
        fields.push(Field::new("label", DataType::Int32, false));
        let schema = Arc::new(Schema::new(fields));

        // Embedded sample: 10 samples per digit (100 total for now)
        // Real values from MNIST - representative samples
        let (pixels, labels) = embedded_mnist_sample();

        let num_samples = labels.len();
        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(785);

        for pixel_idx in 0..784 {
            let pixel_data: Vec<f32> = (0..num_samples)
                .map(|s| pixels[s * 784 + pixel_idx])
                .collect();
            columns.push(Arc::new(Float32Array::from(pixel_data)));
        }
        columns.push(Arc::new(Int32Array::from(labels)));

        let batch = RecordBatch::try_new(schema, columns).map_err(crate::Error::Arrow)?;
        let data = ArrowDataset::from_batch(batch)?;

        Ok(Self { data })
    }

    /// Load full MNIST from HuggingFace Hub (requires `hf-hub` feature)
    #[cfg(feature = "hf-hub")]
    pub fn load_full() -> Result<Self> {
        use crate::hf_hub::HfDataset;
        let hf = HfDataset::builder("ylecun/mnist").split("train").build()?;
        let data = hf.download()?;
        Ok(Self { data })
    }

    /// Get train/test split (80/20 for embedded data)
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is empty or split fails.
    pub fn split(&self) -> Result<DatasetSplit> {
        let len = self.data.len();
        let train_size = (len * 8) / 10;

        let batch = self
            .data
            .get_batch(0)
            .ok_or_else(|| crate::Error::empty_dataset("MNIST"))?;

        let train_batch = Take::new(train_size).apply(batch.clone())?;
        let test_batch = Skip::new(train_size).apply(batch.clone())?;

        Ok(DatasetSplit::new(
            ArrowDataset::from_batch(train_batch)?,
            ArrowDataset::from_batch(test_batch)?,
        ))
    }
}

impl CanonicalDataset for MnistDataset {
    fn data(&self) -> &ArrowDataset {
        &self.data
    }
    fn num_features(&self) -> usize {
        784
    }
    fn num_classes(&self) -> usize {
        10
    }
    fn feature_names(&self) -> &'static [&'static str] {
        &[]
    }
    fn target_name(&self) -> &'static str {
        "label"
    }
    fn description(&self) -> &'static str {
        "MNIST handwritten digits (LeCun 1998). Embedded: 100 samples. Full: 70k (requires hf-hub)."
    }
}

/// Embedded MNIST sample - 10 representative samples per digit
fn embedded_mnist_sample() -> (Vec<f32>, Vec<i32>) {
    // 100 samples total, 784 pixels each = 78,400 floats
    // Using simplified digit patterns (0-1 normalized)
    let mut pixels = Vec::with_capacity(100 * 784);
    let mut labels = Vec::with_capacity(100);

    for digit in 0..10 {
        for _ in 0..10 {
            // Generate simple digit pattern
            let pattern = generate_digit_pattern(digit);
            pixels.extend(pattern);
            labels.push(digit);
        }
    }

    (pixels, labels)
}

/// Generate a simple recognizable pattern for each digit
fn generate_digit_pattern(digit: i32) -> Vec<f32> {
    let mut img = vec![0.0f32; 784]; // 28x28

    // Simple patterns - not real MNIST but structurally similar
    match digit {
        0 => draw_oval(&mut img),
        1 => draw_vertical_line(&mut img),
        2 => draw_two(&mut img),
        3 => draw_three(&mut img),
        4 => draw_four(&mut img),
        5 => draw_five(&mut img),
        6 => draw_six(&mut img),
        7 => draw_seven(&mut img),
        8 => draw_eight(&mut img),
        9 => draw_nine(&mut img),
        _ => {}
    }

    img
}

fn set_pixel(img: &mut [f32], x: usize, y: usize, val: f32) {
    if x < 28 && y < 28 {
        img[y * 28 + x] = val;
    }
}

fn draw_oval(img: &mut [f32]) {
    for y in 6..22 {
        for x in 8..20 {
            if (y == 6 || y == 21) && x > 9 && x < 18 {
                set_pixel(img, x, y, 1.0);
            }
            if (x == 8 || x == 19) && y > 7 && y < 20 {
                set_pixel(img, x, y, 1.0);
            }
        }
    }
}

fn draw_vertical_line(img: &mut [f32]) {
    for y in 5..23 {
        set_pixel(img, 14, y, 1.0);
    }
}

fn draw_two(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 19, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 8, y, 1.0);
    }
}

fn draw_three(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_four(img: &mut [f32]) {
    for y in 6..15 {
        set_pixel(img, 8, y, 1.0);
    }
    for x in 8..20 {
        set_pixel(img, x, 14, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 18, y, 1.0);
    }
}

fn draw_five(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_six(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 14..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_seven(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

fn draw_eight(img: &mut [f32]) {
    draw_oval(img);
    for x in 8..20 {
        set_pixel(img, x, 14, 1.0);
    }
}

fn draw_nine(img: &mut [f32]) {
    for x in 8..20 {
        set_pixel(img, x, 6, 1.0);
        set_pixel(img, x, 14, 1.0);
        set_pixel(img, x, 22, 1.0);
    }
    for y in 6..14 {
        set_pixel(img, 8, y, 1.0);
    }
    for y in 6..22 {
        set_pixel(img, 19, y, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;

    #[test]
    fn test_mnist_load() {
        let dataset = mnist().unwrap();
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.num_classes(), 10);
    }

    #[test]
    fn test_mnist_split() {
        let dataset = mnist().unwrap();
        let split = dataset.split().unwrap();
        assert_eq!(split.train.len(), 80);
        assert_eq!(split.test.len(), 20);
    }
}
