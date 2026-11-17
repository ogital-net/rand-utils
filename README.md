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

## Human-Friendly Strings

```rust
use rand_utils::rand_string_human;

let mut rng = rand::rng();
let s = rand_string_human(&mut rng, 32);
```

Uses a 54-character human-friendly alphabet:
`abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789`.
Excludes `0`, `1`, `i`, `l`, `o`, `I`, `L`, and `O`. Characters are sampled uniformly.
For secrets, supply a cryptographically secure RNG.

To require at least one lowercase letter, one uppercase letter, and one digit:

```rust
use rand_utils::rand_string_human_mixed;

let mut rng = rand::rng();
let s = rand_string_human_mixed(&mut rng, 16);
```

Uses the same alphabet and samples uniformly among strings meeting the requirements.
The length must be at least 3. Panics if none of 256 candidates meets the requirements;
lengths below 3 always panic.

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
