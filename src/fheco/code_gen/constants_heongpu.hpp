#pragma once

#include "fheco/ir/common.hpp"
#include <cstddef>
#include <string_view>
#include <unordered_map>
#include <string>

using namespace std::literals;

namespace fheco::code_gen::heongpu
{

constexpr std::string_view heongpu_namespace{"heongpu"};

// Data types in heongpu are templated on the scheme: Ciphertext<Scheme>, Plaintext<Scheme>
// In the generator we will output e.g. "heongpu::Ciphertext<SCHEME_TYPE>" where SCHEME_TYPE is BFV or CKKS
constexpr std::string_view cipher_type{"heongpu::Ciphertext<SCHEME_TYPE>"};
constexpr std::string_view plain_type{"heongpu::Plaintext<SCHEME_TYPE>"};

constexpr std::string_view header_encrypted_io_type{"std::unordered_map<std::string, heongpu::Ciphertext<SCHEME_TYPE>>"};
constexpr std::string_view header_encoded_io_type{"std::unordered_map<std::string, heongpu::Plaintext<SCHEME_TYPE>>"};

constexpr std::string_view source_encrypted_io_type{"unordered_map<string, heongpu::Ciphertext<SCHEME_TYPE>>"};
constexpr std::string_view source_encoded_io_type{"unordered_map<string, heongpu::Plaintext<SCHEME_TYPE>>"};

constexpr std::string_view encrypted_inputs_container_id{"encrypted_inputs"};
constexpr std::string_view encoded_inputs_container_id{"encoded_inputs"};

constexpr std::string_view encrypted_outputs_container_id{"encrypted_outputs"};
constexpr std::string_view encoded_outputs_container_id{"encoded_outputs"};

constexpr std::string_view relin_keys_type{"heongpu::Relinkey<SCHEME_TYPE>"};
constexpr std::string_view relin_keys_id{"relin_key"};

constexpr std::string_view galois_keys_type{"heongpu::Galoiskey<SCHEME_TYPE>"};
constexpr std::string_view galois_keys_id{"galois_keys"};

constexpr std::string_view context_id{"context"};
constexpr std::string_view ops_id{"ops"};

constexpr std::size_t line_threshold = 16;

constexpr std::string_view header_includes{
  R"(#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <heongpu/heongpu.hpp>
)"};

constexpr std::string_view source_includes{
  R"(#include <cstddef>
#include <cstdint>
#include <utility>
#include <chrono>
#include <iostream>
)"};

constexpr std::string_view source_usings{
  R"(using namespace std;
)"};

// Operation mapping: IR OpCode -> heongpu::HEArithmeticOperator method call
const std::unordered_map<ir::OpType, std::string_view, ir::HashOpType, ir::EqualOpType> operation_mapping = {
  // Cipher + Cipher
  {{ir::OpCode::Type::add, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "add"sv},
  // Cipher + Plain
  {{ir::OpCode::Type::add, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "add_plain"sv},
  
  // Cipher - Cipher
  {{ir::OpCode::Type::sub, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "sub"sv},
  // Cipher - Plain
  {{ir::OpCode::Type::sub, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "sub_plain"sv},
  
  // Cipher * Cipher
  {{ir::OpCode::Type::mul, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "multiply"sv},
  // Cipher * Plain
  {{ir::OpCode::Type::mul, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "multiply_plain"sv},
  
  // Negate
  {{ir::OpCode::Type::negate, {ir::Term::Type::cipher}}, "negate"sv},
  
  // Rotate
  {{ir::OpCode::Type::rotate, {ir::Term::Type::cipher}}, "rotate_rows"sv},
  
  // Relinearize
  {{ir::OpCode::Type::relin, {ir::Term::Type::cipher}}, "relinearize"sv},
  
  // Rescale
  {{ir::OpCode::Type::rescale, {ir::Term::Type::cipher}}, "rescale"sv},
};

// Helper to get the cipher variable name
inline std::string get_cipher_var(std::size_t id) {
  return "c" + std::to_string(id);
}

// Helper to get the plaintext variable name
inline std::string get_plain_var(std::size_t id) {
  return "p" + std::to_string(id);
}

} // namespace fheco::code_gen::heongpu
