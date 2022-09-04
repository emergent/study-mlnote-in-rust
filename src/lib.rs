use smartcore::linalg::naive::dense_matrix::*;

pub fn slope_and_intercept(data: &[(f32, f32)]) -> (f32, f32) {
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

pub fn set_degree(x: f32, deg: usize) -> Vec<f32> {
    let mut v = vec![x];
    for _ in 1..deg {
        v.push(v[v.len() - 1] * x);
    }
    v
}

pub fn format_legend(
    deg: usize,
    coef: &DenseMatrix<f32>,
    intercept: f32,
) -> String {
    let mut s = String::new();
    s = format!("{}{}", s, intercept);
    for i in 0..deg {
        s = format!("{} + {}x^{}", s, coef.get(i, 0), i + 1);
    }
    s
}
