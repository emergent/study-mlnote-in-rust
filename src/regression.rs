use anyhow::Result;
use derive_new::new;
use plotters::prelude::*;
use smartcore::linalg::naive::dense_matrix::*;
use smartcore::linear::linear_regression::*;
use std::ops::{Range, RangeBounds};

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

pub fn set_degree(x: f32, degree: usize) -> Vec<f32> {
    let mut v = vec![x];
    for _ in 1..degree {
        v.push(v[v.len() - 1] * x);
    }
    v
}

#[derive(new)]
pub struct Polynomial {
    lr: LinearRegression<f32, DenseMatrix<f32>>,
    degree: usize,
}

impl Polynomial {
    pub fn polyfit(
        x: &[f32],
        y: &[f32],
        degree: usize,
    ) -> Result<Self> {
        let x = DenseMatrix::from_2d_vec(
            &x.iter()
                .map(|&x| set_degree(x, degree))
                .collect::<Vec<Vec<_>>>(),
        );
        let y = y.to_vec();

        let lr = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters::default()
                .with_solver(LinearRegressionSolverName::QR),
        )?;

        Ok(Self { lr, degree })
    }

    pub fn polyval<R>(&self, range: R) -> Result<Vec<f32>>
    where
        R: RangeBounds<i32> + Iterator<Item = i32>,
    {
        let x = range
            .map(|x| x as f32)
            .map(|x| set_degree(x, self.degree))
            .collect::<Vec<Vec<_>>>();

        let dm = DenseMatrix::from_2d_vec(&x);
        let yhat = self.lr.predict(&dm)?;
        Ok(yhat)
    }

    pub fn format_legend(&self) -> String {
        let mut s = String::new();
        s = format!("{}{}", s, self.lr.intercept());
        for i in 0..self.degree {
            let coef = self.lr.coefficients().get(i, 0);
            let (op, coef) = if coef >= 0.0 {
                ('+', coef)
            } else {
                ('-', -coef)
            };

            s = format!("{} {} {}x^{}", s, op, coef, i + 1);
        }
        s
    }
}

#[derive(new)]
pub struct PlotConfig<'a> {
    data: Vec<(f32, f32)>,
    x_range: Range<f32>,
    y_range: Range<f32>,
    caption: &'a str,
    x_desc: &'a str,
    y_desc: &'a str,
    with_coord: bool,
}

impl<'a> PlotConfig<'a> {
    pub fn data(&self) -> &[(f32, f32)] {
        &self.data
    }
}

pub fn config_simple<'a>() -> PlotConfig<'a> {
    PlotConfig::new(
        vec![(1., 3.), (3., 6.), (6., 5.), (8., 7.)],
        0f32..10f32,
        0f32..10f32,
        "",
        "x",
        "y",
        true,
    )
}

pub fn config_icecream<'a>() -> PlotConfig<'a> {
    PlotConfig::new(
        std::iter::zip(
            dataset::TEMPERATURES.to_vec(),
            dataset::SPENDINGS.to_vec(),
        )
        .collect::<Vec<_>>(),
        0f32..35f32,
        -250f32..2000f32,
        "Highest temperatures and ice cream/sorbet spending",
        "Monthly avg of max temperature (Celsius)",
        "Spending (Yen)",
        false,
    )
}

pub fn plot<P: AsRef<std::path::Path>>(
    path: P,
    config: PlotConfig<'_>,
    series: Vec<(f32, f32)>,
    legend: &str,
) -> Result<()> {
    let root =
        BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(20, 20, 20, 20);

    let mut chart = ChartBuilder::on(&root)
        .caption(config.caption, ("sans-serif", 30).into_font())
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(config.x_range, config.y_range)?;

    chart
        .configure_mesh()
        .x_desc(config.x_desc)
        .y_desc(config.y_desc)
        .axis_desc_style(("sans-serif", 30).into_font())
        .max_light_lines(0)
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{}", *x))
        .y_label_formatter(&|y| format!("{}", *y))
        .label_style(("sans-serif", 20).into_font())
        .draw()?;

    // plot data
    if config.with_coord {
        chart.draw_series(PointSeries::of_element(
            config.data.clone(),
            5,
            &BLUE,
            &|c, s, st| {
                EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled())
                    + Text::new(
                        format!("({}, {})", c.0, c.1),
                        (-15, -25),
                        ("sans-serif", 20).into_font(),
                    )
            },
        ))?;
    } else {
        chart.draw_series(
            config
                .data
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())),
        )?;
    };

    // plot linear regression
    chart
        .draw_series(LineSeries::new(series, &RED))?
        .label(legend)
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &RED)
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 20))
        .draw()?;

    root.present()?;

    Ok(())
}

pub mod dataset {
    pub const TEMPERATURES: &[f32] = &[
        9.1, 11.2, 12.3, 18.9, 22.2, 26., 30.9, 31.2, 28.8, 23., 18.3,
        11.1, 8.3, 9.1, 12.5, 18.5, 23.6, 24.8, 30.1, 33.1, 29.8, 23.,
        16.3, 11.2, 9.6, 10.3, 16.4, 19.2, 24.1, 26.5, 31.4, 33.2,
        28.8, 23., 17.4, 12.1, 10.6, 9.8, 14.5, 19.6, 24.7, 26.9, 30.5,
        31.2, 26.9, 23., 17.4, 11., 10.4, 10.4, 15.5, 19.3, 26.4, 26.4,
        30.1, 30.5, 26.4, 22.7, 17.8, 13.4, 10.6, 12.2, 14.9, 20.3,
        25.2, 26.3, 29.7, 31.6, 27.7, 22.6, 15.5, 13.8, 10.8, 12.1,
        13.4, 19.9, 25.1, 26.4, 31.8, 30.4, 26.8, 20.1, 16.6, 11.1,
        9.4, 10.1, 16.9, 22.1, 24.6, 26.6, 32.7, 32.5, 26.6, 23., 17.7,
        12.1, 10.3, 11.6, 15.4, 19., 25.3, 25.8, 27.5, 32.8, 29.4,
        23.3, 17.7, 12.6, 11.1, 13.3, 16., 18.2, 24., 27.5, 27.7, 34.1,
        28.1, 21.4, 18.6, 12.3,
    ];

    pub const SPENDINGS: &[f32] = &[
        463., 360., 380., 584., 763., 886., 1168., 1325., 847., 542.,
        441., 499., 363., 327., 414., 545., 726., 847., 1122., 1355.,
        916., 571., 377., 465., 377., 362., 518., 683., 838., 1012.,
        1267., 1464., 1000., 629., 448., 466., 404., 343., 493., 575.,
        921., 1019., 1149., 1303., 805., 739., 587., 561., 486., 470.,
        564., 609., 899., 946., 1295., 1325., 760., 667., 564., 633.,
        478., 450., 567., 611., 947., 962., 1309., 1307., 930., 668.,
        496., 650., 506., 423., 531., 672., 871., 986., 1368., 1319.,
        924., 716., 651., 708., 609., 535., 717., 890., 1054., 1077.,
        1425., 1378., 900., 725., 554., 542., 561., 459., 604., 745.,
        1105., 973., 1263., 1533., 1044., 821., 621., 601., 549., 572.,
        711., 819., 1141., 1350., 1285., 1643., 1133., 784., 682.,
        587.,
    ];
}
