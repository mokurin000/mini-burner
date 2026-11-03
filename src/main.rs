use std::error::Error;

use mini_burner::{preferred_execution_provider, timer::Timer};
use ndarray::Array;
use ort::{inputs, session::Session, value::Tensor};
use webp::Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let repeat = 16;
    let shape = vec![repeat, 3, 640, 640];

    let input_data = {
        let _timer = Timer::with_tip("Prepared single input tensor");
        prepare_rgb_data()
    };
    let array = {
        let _timer = Timer::with_tip("Prepared batched input tensor");
        Array::from_shape_vec(shape, input_data.repeat(repeat))?
    };

    let env = ort::init().build()?;

    let mut session = {
        let _timer = Timer::with_tip("Loaded the classification model");
        Session::builder(&env)?
            .with_execution_providers([preferred_execution_provider()])?
            .commit_from_file("model/yolo26l-cls.onnx")?
    };

    loop {
        let input_tensor = Tensor::from_array(array.clone())?;
        let _output = {
            let _timer = Timer::with_tip("Batch inferenced the classification");
            session.run(inputs!["images" => input_tensor])?
        };
    }
}

fn prepare_rgb_data() -> Vec<f32> {
    let picture = include_bytes!("image/cat.webp");

    // decode the bundled tabby image
    let Some(cat_pic) = Decoder::new(picture).decode() else {
        unreachable!("bundled image must be valid webp")
    };
    let rgb = if cat_pic.is_alpha() {
        cat_pic.chunks_exact(4)
    } else {
        cat_pic.chunks_exact(3)
    }
    .flatten()
    .map(|pixel| *pixel as f32 / 255.)
    .collect::<Vec<_>>();

    // split *[RGB] into *[R] *[G] *[B]
    let (mut r, mut g, mut b) =
        rgb.chunks_exact(3)
            .fold((vec![], vec![], vec![]), |mut rgbs, rgb| {
                rgbs.0.push(rgb[0]);
                rgbs.1.push(rgb[1]);
                rgbs.2.push(rgb[2]);
                rgbs
            });

    // merge single color channel images
    let mut input_data = Vec::new();
    input_data.append(&mut r);
    input_data.append(&mut g);
    input_data.append(&mut b);

    input_data
}
