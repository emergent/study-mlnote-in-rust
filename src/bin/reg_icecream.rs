use anyhow::Result;
use mlnote_rs::{get_output_path, regression::*};
use std::path::Path;

fn main() -> Result<()> {
    let path = get_output_path();
    run(&path)?;
    Ok(())
}

fn run<P: AsRef<Path>>(path: P) -> Result<()> {
    let config = config_icecream();

    let (a, b) = slope_and_intercept(config.data());
    let series = (0..=35)
        .map(|x| (x as f32, a * (x as f32) + b))
        .collect::<Vec<_>>();
    let legend = format!("{:.3}x + {:.3}", a, b);

    plot(path, config, series, &legend)?;

    Ok(())
}
