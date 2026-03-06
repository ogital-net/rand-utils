# rand-utils

Random string generation with SIMD optimizations.

## Usage

```rust
use rand_utils::rand_string;

let mut rng = rand::rng();
let s = rand_string(&mut rng, 32);
```

Generates strings from the charset: `A-Z`, `a-z`, `0-9`, `-`, `_`

Uses SSSE3 on x86_64 and NEON on ARM when available.

## AWS-LC RNG

Optional cryptographically secure RNG backed by AWS-LC:

```toml
[dependencies]
rand-utils = { version = "0.1", features = ["aws-lc"] }
```

```rust
use rand_utils::AwsLcRng;

let mut rng = AwsLcRng;
let s = rand_string(&mut rng, 32);
```

## Requirements

Rust 1.84+
