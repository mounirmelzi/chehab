#pragma once

#include "fheco/ir/common.hpp"
#include <cstddef>
#include <string_view>
#include <unordered_map>

using namespace std::literals;

namespace fheco::code_gen::lattigo
{

// Package and import paths
constexpr std::string_view lattigo_ckks_import{"github.com/tuneinsight/lattigo/v5/he/hefloat"};
constexpr std::string_view lattigo_rlwe_import{"github.com/tuneinsight/lattigo/v5/core/rlwe"};

// Type names in Lattigo
constexpr std::string_view cipher_type{"*rlwe.Ciphertext"};
constexpr std::string_view plain_type{"*rlwe.Plaintext"};
constexpr std::string_view evaluator_type{"*hefloat.Evaluator"};
constexpr std::string_view encoder_type{"*hefloat.Encoder"};
constexpr std::string_view encryptor_type{"*rlwe.Encryptor"};
constexpr std::string_view decryptor_type{"*rlwe.Decryptor"};
constexpr std::string_view params_type{"hefloat.Parameters"};
constexpr std::string_view secret_key_type{"*rlwe.SecretKey"};
constexpr std::string_view public_key_type{"*rlwe.PublicKey"};
constexpr std::string_view relin_key_type{"*rlwe.RelinearizationKey"};
constexpr std::string_view galois_key_type{"*rlwe.GaloisKey"};
constexpr std::string_view eval_key_type{"*rlwe.MemEvaluationKeySet"};

// Variable identifiers
constexpr std::string_view evaluator_id{"eval"};
constexpr std::string_view encoder_id{"encoder"};
constexpr std::string_view encryptor_id{"enc"};
constexpr std::string_view decryptor_id{"dec"};
constexpr std::string_view params_id{"params"};
constexpr std::string_view secret_key_id{"sk"};
constexpr std::string_view public_key_id{"pk"};
constexpr std::string_view relin_key_id{"rlk"};
constexpr std::string_view galois_key_id{"gks"};
constexpr std::string_view eval_key_id{"evk"};

// Container identifiers
constexpr std::string_view encrypted_inputs_id{"encryptedInputs"};
constexpr std::string_view encoded_inputs_id{"encodedInputs"};
constexpr std::string_view encrypted_outputs_id{"encryptedOutputs"};
constexpr std::string_view encoded_outputs_id{"encodedOutputs"};

// Go file header with imports (without bootstrapping) - "time" added back for benchmarks
constexpr std::string_view go_file_header{
R"(package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

)"};

// Go file header with imports (with bootstrapping - Orion compatible) - "time" added back
constexpr std::string_view go_file_header_bootstrap{
R"(package main

import (
	"fmt"
	"math"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

// Global bootstrapper (will be initialized in main)
var bootstrapper *bootstrapping.Evaluator

)"};

// Go main function template for CKKS
constexpr std::string_view go_main_template{
R"(
func main() {
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galoisElements, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	_ = encoder
	_ = enc
	_ = dec
	_ = eval
	_ = encryptedInputs
	_ = encodedInputs
	_ = encryptedOutputs
	_ = encodedOutputs

	fmt.Println("CKKS computation completed successfully!")
}
)"};

const std::unordered_map<ir::OpType, std::string_view, ir::HashOpType, ir::EqualOpType> operation_mapping = {
  {{ir::OpCode::Type::add, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "Add"sv},
  {{ir::OpCode::Type::add, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "Add"sv},
  {{ir::OpCode::Type::sub, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "Sub"sv},
  {{ir::OpCode::Type::sub, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "Sub"sv},
  {{ir::OpCode::Type::negate, {ir::Term::Type::cipher}}, "Neg"sv},
  {{ir::OpCode::Type::rotate, {ir::Term::Type::cipher}}, "Rotate"sv},
  {{ir::OpCode::Type::square, {ir::Term::Type::cipher}}, "MulRelin"sv},
  {{ir::OpCode::Type::mul, {ir::Term::Type::cipher, ir::Term::Type::cipher}}, "MulRelin"sv},
  {{ir::OpCode::Type::mul, {ir::Term::Type::cipher, ir::Term::Type::plain}}, "Mul"sv},
  {{ir::OpCode::Type::mod_switch, {ir::Term::Type::cipher}}, "DropLevel"sv},
  {{ir::OpCode::Type::relin, {ir::Term::Type::cipher}}, "Relinearize"sv},
  {{ir::OpCode::Type::rescale, {ir::Term::Type::cipher}}, "Rescale"sv},
  {{ir::OpCode::Type::bootstrap, {ir::Term::Type::cipher}}, "Bootstrap"sv},
  {{ir::OpCode::Type::SumVec, {ir::Term::Type::cipher}}, "SumVec"sv},
};

inline std::string get_cipher_var(std::size_t id) {
  return "c" + std::to_string(id);
}

inline std::string get_plain_var(std::size_t id) {
  return "p" + std::to_string(id);
}

} // namespace fheco::code_gen::lattigo