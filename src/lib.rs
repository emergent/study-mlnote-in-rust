pub mod regression;

use std::path::{Path, PathBuf};

const OUTDIR: &str = "plotters-images";

pub fn get_output_path() -> PathBuf {
    let arg = std::env::args().next().unwrap();
    let fname = Path::new(&arg).file_name().unwrap().to_string_lossy();
    let plot_image_filename = format!("{}.png", fname);
    PathBuf::from(OUTDIR).join(plot_image_filename)
}
