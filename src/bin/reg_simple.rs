use anyhow::Result;
use mlnote_rs::{get_output_path, regression::*};
use std::path::Path;

fn main() -> Result<()> {
    let path = get_output_path();
    run(&path)?;
    Ok(())
}

fn run<P: AsRef<Path>>(path: P) -> Result<()> {
    let config = config_simple();

    let (a, b) = slope_and_intercept(config.data());
    let series = (0..=10)
        .map(|x| (x as f32, a * (x as f32) + b))
        .collect::<Vec<_>>();
    let legend = format!("{:.3}x + {:.3}", a, b);

    plot(path, config, series, &legend)?;

    Ok(())
}

fn slope_and_intercept(data: &[(f32, f32)]) -> (f32, f32) {
    let len = data.len() as f32;
    let x_avg = data.iter().map(|d| d.0).sum::<f32>() / len;
    let y_avg = data.iter().map(|d| d.1).sum::<f32>() / len;

    let cov_xy = data
        .iter()
        .map(|&(x, y)| (x - x_avg) * (y - y_avg))
        .sum::<f32>()
        / len;
    let var_x = data
        .iter()
        .map(|&(x, _)| (x - x_avg) * (x - x_avg))
        .sum::<f32>()
        / len;

    let a = cov_xy / var_x;
    (a, y_avg - a * x_avg)
}
