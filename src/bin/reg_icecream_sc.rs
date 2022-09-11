use anyhow::Result;
use mlnote_rs::{get_output_path, regression::*};
use smartcore::linalg::naive::dense_matrix::*;
use smartcore::linear::linear_regression::*;
use std::path::Path;

fn main() -> Result<()> {
    let path = get_output_path();
    run(&path)?;
    Ok(())
}

fn run<P: AsRef<Path>>(path: P) -> Result<()> {
    let config = config_icecream();

    let x = DenseMatrix::from_2d_vec(
        &TEMPERATURES
            .iter()
            .map(|x| vec![*x])
            .collect::<Vec<Vec<f32>>>(),
    );
    let y = SPENDINGS.to_vec();

    let lr = LinearRegression::fit(
        &x,
        &y,
        LinearRegressionParameters::default()
            .with_solver(LinearRegressionSolverName::QR),
    )?;
    let x = (0..=35).map(|x| vec![x as f32]).collect::<Vec<Vec<_>>>();
    let dm = DenseMatrix::from_2d_vec(&x);
    let yhat = lr.predict(&dm)?;
    let coef = lr.coefficients().get(0, 0);
    let intercept = lr.intercept();

    let series =
        std::iter::zip(x.into_iter().flatten().collect::<Vec<_>>(), yhat)
            .collect::<Vec<_>>();
    let legend = format!("{:.3}x + {:.3}", coef, intercept);

    plot(path, config, series, &legend)?;

    Ok(())
}
