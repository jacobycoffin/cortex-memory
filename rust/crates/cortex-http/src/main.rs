fn main() {
    let code = cortex_http::run(std::env::args_os());
    std::process::exit(code);
}
