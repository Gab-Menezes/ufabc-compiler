# Projeto Matéria de Compiladores UFABC (2024)

## [Link vídeo](https://youtu.be/JvIuzgGkmOs)

## Gabriel Jorge Menezes: 11201921315

## Running

`cargo run --release -- -i {input.isi} -o {output}`

## Generating [test_programs](./test_programs)

* `cargo run --release -- -i ./test_programs/empty.isi -o ./test_programs/empty`
* `cargo run --release -- -i ./test_programs/var_declaration.isi -o ./test_programs/var_declaration`
* `cargo run --release -- -i ./test_programs/structures.isi -o ./test_programs/structures`
* `cargo run --release -- -i ./test_programs/fib.isi -o ./test_programs/fib`

* `cargo run --release -- -i ./test_programs/error_type_mismatch.isi -o ./test_programs/error_type_mismatch`
* `cargo run --release -- -i ./test_programs/error_variable_already_declared.isi -o ./test_programs/error_variable_already_declared`
* `cargo run --release -- -i ./test_programs/error_variable_not_declared.isi -o ./test_programs/error_variable_not_declared`
* `cargo run --release -- -i ./test_programs/error_variable_not_used.isi -o ./test_programs/error_variable_not_used`
* `cargo run --release -- -i ./test_programs/error_invalid_operation.isi -o ./test_programs/error_invalid_operation`