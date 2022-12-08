pub fn compile_shader() -> String {
  wgsl_preprocessor::ShaderBuilder::new("./src/kernel/main.wgsl")
    .expect("Failed to load ./src/kernel/main.wgsl")
    .put_array_definition("hutton32_colors",
      &HUTTON32_COLORS.map(|x| x.map(|c| c as i32))
        .iter().collect()
    ).source_string.clone()
}

pub const HUTTON32_COLORS: [[u8; 3]; 32] = [
  [ 0 ,   0,   0],    // 0  dark gray
  [255,   0,   0],    // 1  red
  [255, 125,   0],    // 2  orange (to match red and yellow)
  [255, 150,  25],    // 3   lighter
  [255, 175,  50],    // 4    lighter
  [255, 200,  75],    // 5     lighter
  [255, 225, 100],    // 6      lighter
  [255, 250, 125],    // 7       lighter
  [251, 255,   0],    // 8  yellow
  [ 89,  89, 255],    // 9  blue
  [106, 106, 255],    // 10  lighter
  [122, 122, 255],    // 11   lighter
  [139, 139, 255],    // 12    lighter
  [ 27, 176,  27],    // 13 green
  [ 36, 200,  36],    // 14  lighter
  [ 73, 255,  73],    // 15   lighter
  [106, 255, 106],    // 16    lighter
  [235,  36,  36],    // 17 red
  [255,  56,  56],    // 18  lighter
  [255,  73,  73],    // 19   lighter
  [255,  89,  89],    // 20    lighter
  [185,  56, 255],    // 21 purple
  [191,  73, 255],    // 22  lighter
  [197,  89, 255],    // 23   lighter
  [203, 106, 255],    // 24    lighter
  [  0, 255, 128],    // 25 light green
  [255, 128,  64],    // 26 light orange
  [255, 255, 128],    // 27 light yellow
  [ 33, 215, 215],    // 28 cyan
  [ 27, 176, 176],    // 29  darker
  [ 24, 156, 156],    // 30   darker
  [ 21, 137, 137]     // 31    darker
];