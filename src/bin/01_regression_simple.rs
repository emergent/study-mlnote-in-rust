use anyhow::Result;
use plotters::prelude::*;
use std::path::Path;

const OUTDIR: &str = "plotters-images";

fn main() -> Result<()> {
    let arg = std::env::args().next().unwrap();
    let plot_image_filename =
        Path::new(&arg).file_name().unwrap().to_string_lossy();
    let path = format!("{}/{}.png", OUTDIR, plot_image_filename);
    plot(&path)?;
    Ok(())
}

fn plot(filename: &str) -> Result<()> {
    let data: Vec<(f32, f32)> =
        vec![(1., 3.), (3., 6.), (6., 5.), (8., 7.)];

    let x_range = 0f32..10f32;
    let y_range = 0f32..10f32;
    let caption = "";
    let x_desc = "x";
    let y_desc = "y";

    let root =
        BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(20, 20, 20, 20);

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 30).into_font())
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .axis_desc_style(("sans-serif", 30).into_font())
        .max_light_lines(0)
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{}", *x))
        .y_label_formatter(&|y| format!("{}", *y))
        .label_style(("sans-serif", 20).into_font())
        .draw()?;

    chart.draw_series(PointSeries::of_element(
        data.clone(),
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

    // plot linear regression
    let (a, b) = slope_and_intercept(&data);
    chart
        .draw_series(LineSeries::new(
            (0..=10).map(|x| (x as f32, a * (x as f32) + b)),
            &BLACK,
        ))?
        .label(&format!("{:.3}x + {:.3}", a, b))
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &RED)
        });

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
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
