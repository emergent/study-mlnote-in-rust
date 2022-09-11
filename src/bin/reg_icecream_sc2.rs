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

    // plot linear regression
    const DEGREE: usize = 2;
    let x = DenseMatrix::from_2d_vec(
        &TEMPERATURES
            .iter()
            .map(|&x| set_degree(x, DEGREE))
            .collect::<Vec<Vec<f32>>>(),
    );
    let y = SPENDINGS.to_vec();

    let lr = LinearRegression::fit(
        &x,
        &y,
        LinearRegressionParameters::default()
            .with_solver(LinearRegressionSolverName::QR),
    )?;
    let x = (0..=35)
        .map(|x| x as f32)
        .map(|x| set_degree(x, DEGREE))
        .collect::<Vec<Vec<_>>>();
    let dm = DenseMatrix::from_2d_vec(&x);
    let yhat = lr.predict(&dm)?;
    let coef = lr.coefficients();
    let intercept = lr.intercept();
    let legend = format_legend(DEGREE, coef, intercept);

    let series = std::iter::zip(
        x.into_iter().map(|x| x[0]).collect::<Vec<_>>(),
        yhat,
    )
    .collect::<Vec<_>>();

    plot(path, config, series, &legend)?;

    Ok(())
}
