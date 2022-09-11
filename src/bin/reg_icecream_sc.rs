use anyhow::Result;
use mlnote_rs::{get_output_path, regression::*};
use std::{ops::RangeInclusive, path::Path};

const DEGREE: usize = 1;
const RANGE: RangeInclusive<i32> = 0..=35;

fn main() -> Result<()> {
    let path = get_output_path();
    run(&path)?;
    Ok(())
}

fn run<P: AsRef<Path>>(path: P) -> Result<()> {
    let config = config_icecream();

    let pn = Polynomial::polyfit(
        dataset::TEMPERATURES,
        dataset::SPENDINGS,
        DEGREE,
    )?;
    let yhat = pn.polyval(RANGE)?;
    let legend = pn.format_legend();

    let series =
        std::iter::zip(RANGE.map(|x| x as f32), yhat).collect::<Vec<_>>();

    plot(path, config, series, &legend)?;

    Ok(())
}
