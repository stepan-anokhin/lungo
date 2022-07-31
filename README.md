# Lungo Programming Language

`Lungo` is a dynamic programming language created just for fun.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/stepan-anokhin/lungo/blob/master/LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/lungo.svg)](https://pypi.org/project/lungo/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lungo.svg)](https://pypi.org/project/lungo/)

## Getting Started

Installations:

```shell
pip install --upgrade lungo
```

Usage:

```shell
lung [FILE]
```

## Syntax

You need to define variable with `let` statement before assignment/reference:

```
let x = 2;
```

Assign a previously defined variable:

```
let x = 0;
x = 42;
```

Define a function:

```
func inc(x) {
    return x + 1
}
```

The last statement in each code block is a result of the block evaluation, so in the previous example `return` is not
necessary:

```
func inc(x) {
  x + 1
}
```

Function definitions are just an expressions. The previous line is equivalent to:

```
let inc = func(x) { x + 1 }
```

Conditional execution:

```
if ( x > 0 ) {
  print("Positive")
} elif ( x < 0 ) {
  print("Negative")
} else {
  print("X is 0")
}
```

`while`-loop:

```
let x = 0;
while ( x < 10 ) {
  x = x + 1;
  print(x)
}
```

`for`-loop:

```
for ( x in [1,2,3] ) {
  print(x)
}
```

Almost everything is expression in Lungo (except for `return` and `let` statements):

```
let x = 10;
let message = if(x > 5) { "big" } else { "small" };
print(message)
```

Output:

```
big
```

Another example:

```
func hello(value) {
    print("Hello " + value)
};

let who = "world";

hello(while(who.size < 20) { who = who + " and " + who })
```

Output:

```
Hello world and world and world and world
```

**NOTE**: A bit of syntactic bitterness. In the current implementation each statement in a code block (except for the
last one) must be followed by a semicolon `;`. So the `;` must be present after such expressions as function
definitions, `for` and `while`-loops, etc. This will be fixed in a future release.

Supported binary operators: `+`, `-`, `*`, `/`, `&&`, `||`, `==`, `!=`, `>`, `>=`, `<`, `<=`

Unary operators: `!`, `-`

Higher order functions:

```
func inc(x) {
  x + 1
};

func comp(f, g) {
  func(x) {
    f(g(x))
  }
}

print(comp(inc, inc)(0))
```

Output:

```
2
```

More sophisticated example:

```
func pair(a, b) {
    func(acceptor) {
        acceptor(a, b)
    }
};

func first(p) {
    p(func(a,b) { a })
};

func second(p) {
    p(func(a,b) { b })
};

let p = pair(1, 2);

print(first(p));
print(second(p));
```

Output:

```
1
2
```
