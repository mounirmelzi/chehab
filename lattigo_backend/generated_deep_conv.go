package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func getRotationSteps() []int {
	return []int{}
}

func deep_conv(
	encryptedInputs map[string]*rlwe.Ciphertext,
	encodedInputs map[string]*rlwe.Plaintext,
	encryptedOutputs map[string]*rlwe.Ciphertext,
	encodedOutputs map[string]*rlwe.Plaintext,
	encoder *hefloat.Encoder,
	enc *rlwe.Encryptor,
	eval *hefloat.Evaluator,
	params hefloat.Parameters,
) {
	c29 := encryptedInputs["img_4_4"]
	c417 := encryptedInputs["k2_1_0"]
	c28 := encryptedInputs["img_4_3"]
	c416 := encryptedInputs["k2_0_1"]
	c40 := encryptedInputs["k0_1_1"]
	c27 := encryptedInputs["img_4_2"]
	c415 := encryptedInputs["k2_0_0"]
	c39 := encryptedInputs["k0_1_0"]
	c38 := encryptedInputs["k0_0_1"]
	c37 := encryptedInputs["k0_0_0"]
	c36 := encryptedInputs["img_5_5"]
	c270 := encryptedInputs["k1_1_1"]
	c35 := encryptedInputs["img_5_4"]
	c269 := encryptedInputs["k1_1_0"]
	c34 := encryptedInputs["img_5_3"]
	c268 := encryptedInputs["k1_0_1"]
	c33 := encryptedInputs["img_5_2"]
	c267 := encryptedInputs["k1_0_0"]
	c32 := encryptedInputs["img_5_1"]
	c31 := encryptedInputs["img_5_0"]
	c418 := encryptedInputs["k2_1_1"]
	c30 := encryptedInputs["img_4_5"]
	c26 := encryptedInputs["img_4_1"]
	c25 := encryptedInputs["img_4_0"]
	c24 := encryptedInputs["img_3_5"]
	c23 := encryptedInputs["img_3_4"]
	c22 := encryptedInputs["img_3_3"]
	c21 := encryptedInputs["img_3_2"]
	c20 := encryptedInputs["img_3_1"]
	c19 := encryptedInputs["img_3_0"]
	c18 := encryptedInputs["img_2_5"]
	c17 := encryptedInputs["img_2_4"]
	c16 := encryptedInputs["img_2_3"]
	c15 := encryptedInputs["img_2_2"]
	c14 := encryptedInputs["img_2_1"]
	c13 := encryptedInputs["img_2_0"]
	c12 := encryptedInputs["img_1_5"]
	c11 := encryptedInputs["img_1_4"]
	c10 := encryptedInputs["img_1_3"]
	c9 := encryptedInputs["img_1_2"]
	c8 := encryptedInputs["img_1_1"]
	c7 := encryptedInputs["img_1_0"]
	c6 := encryptedInputs["img_0_5"]
	c5 := encryptedInputs["img_0_4"]
	c4 := encryptedInputs["img_0_3"]
	c3 := encryptedInputs["img_0_2"]
	c2 := encryptedInputs["img_0_1"]
	c1 := encryptedInputs["img_0_0"]

	// FHE Operations
	c1, _ = eval.MulRelinNew(c1, c37)
	var c735 *rlwe.Ciphertext
	_ = eval.Relinearize(c1, c735)
	var c44 *rlwe.Ciphertext
	c44, _ = eval.MulRelinNew(c2, c38)
	var c736 *rlwe.Ciphertext
	_ = eval.Relinearize(c44, c736)
	c735, _ = eval.AddNew(c735, c736)
	c736, _ = eval.MulRelinNew(c7, c39)
	var c737 *rlwe.Ciphertext
	_ = eval.Relinearize(c736, c737)
	c735, _ = eval.AddNew(c735, c737)
	c737, _ = eval.MulRelinNew(c8, c40)
	var c738 *rlwe.Ciphertext
	_ = eval.Relinearize(c737, c738)
	c735, _ = eval.AddNew(c735, c738)
	c735, _ = eval.MulRelinNew(c735, c735)
	_ = eval.Relinearize(c735, c738)
	c738, _ = eval.MulRelinNew(c738, c267)
	var c740 *rlwe.Ciphertext
	_ = eval.Relinearize(c738, c740)
	c2, _ = eval.MulRelinNew(c2, c37)
	var c720 *rlwe.Ciphertext
	_ = eval.Relinearize(c2, c720)
	var c53 *rlwe.Ciphertext
	c53, _ = eval.MulRelinNew(c3, c38)
	var c721 *rlwe.Ciphertext
	_ = eval.Relinearize(c53, c721)
	c720, _ = eval.AddNew(c720, c721)
	c721, _ = eval.MulRelinNew(c8, c39)
	var c722 *rlwe.Ciphertext
	_ = eval.Relinearize(c721, c722)
	c720, _ = eval.AddNew(c720, c722)
	c722, _ = eval.MulRelinNew(c9, c40)
	var c723 *rlwe.Ciphertext
	_ = eval.Relinearize(c722, c723)
	c720, _ = eval.AddNew(c720, c723)
	c720, _ = eval.MulRelinNew(c720, c720)
	_ = eval.Relinearize(c720, c723)
	var c273 *rlwe.Ciphertext
	c273, _ = eval.MulRelinNew(c723, c268)
	var c741 *rlwe.Ciphertext
	_ = eval.Relinearize(c273, c741)
	c740, _ = eval.AddNew(c740, c741)
	c7, _ = eval.MulRelinNew(c7, c37)
	_ = eval.Relinearize(c7, c741)
	var c89 *rlwe.Ciphertext
	c89, _ = eval.MulRelinNew(c8, c38)
	var c676 *rlwe.Ciphertext
	_ = eval.Relinearize(c89, c676)
	c741, _ = eval.AddNew(c741, c676)
	c676, _ = eval.MulRelinNew(c13, c39)
	var c677 *rlwe.Ciphertext
	_ = eval.Relinearize(c676, c677)
	c741, _ = eval.AddNew(c741, c677)
	c677, _ = eval.MulRelinNew(c14, c40)
	var c678 *rlwe.Ciphertext
	_ = eval.Relinearize(c677, c678)
	c741, _ = eval.AddNew(c741, c678)
	c741, _ = eval.MulRelinNew(c741, c741)
	_ = eval.Relinearize(c741, c678)
	var c275 *rlwe.Ciphertext
	c275, _ = eval.MulRelinNew(c678, c269)
	var c742 *rlwe.Ciphertext
	_ = eval.Relinearize(c275, c742)
	c740, _ = eval.AddNew(c740, c742)
	c8, _ = eval.MulRelinNew(c8, c37)
	_ = eval.Relinearize(c8, c742)
	var c98 *rlwe.Ciphertext
	c98, _ = eval.MulRelinNew(c9, c38)
	var c661 *rlwe.Ciphertext
	_ = eval.Relinearize(c98, c661)
	c742, _ = eval.AddNew(c742, c661)
	c661, _ = eval.MulRelinNew(c14, c39)
	var c662 *rlwe.Ciphertext
	_ = eval.Relinearize(c661, c662)
	c742, _ = eval.AddNew(c742, c662)
	c662, _ = eval.MulRelinNew(c15, c40)
	var c663 *rlwe.Ciphertext
	_ = eval.Relinearize(c662, c663)
	c742, _ = eval.AddNew(c742, c663)
	c742, _ = eval.MulRelinNew(c742, c742)
	_ = eval.Relinearize(c742, c663)
	var c277 *rlwe.Ciphertext
	c277, _ = eval.MulRelinNew(c663, c270)
	var c743 *rlwe.Ciphertext
	_ = eval.Relinearize(c277, c743)
	c740, _ = eval.AddNew(c740, c743)
	c740, _ = eval.MulRelinNew(c740, c740)
	_ = eval.Relinearize(c740, c743)
	c743, _ = eval.MulRelinNew(c743, c415)
	var c745 *rlwe.Ciphertext
	_ = eval.Relinearize(c743, c745)
	c723, _ = eval.MulRelinNew(c723, c267)
	var c725 *rlwe.Ciphertext
	_ = eval.Relinearize(c723, c725)
	c3, _ = eval.MulRelinNew(c3, c37)
	var c690 *rlwe.Ciphertext
	_ = eval.Relinearize(c3, c690)
	var c62 *rlwe.Ciphertext
	c62, _ = eval.MulRelinNew(c4, c38)
	var c691 *rlwe.Ciphertext
	_ = eval.Relinearize(c62, c691)
	c690, _ = eval.AddNew(c690, c691)
	c691, _ = eval.MulRelinNew(c9, c39)
	var c692 *rlwe.Ciphertext
	_ = eval.Relinearize(c691, c692)
	c690, _ = eval.AddNew(c690, c692)
	c692, _ = eval.MulRelinNew(c10, c40)
	var c693 *rlwe.Ciphertext
	_ = eval.Relinearize(c692, c693)
	c690, _ = eval.AddNew(c690, c693)
	c690, _ = eval.MulRelinNew(c690, c690)
	_ = eval.Relinearize(c690, c693)
	var c282 *rlwe.Ciphertext
	c282, _ = eval.MulRelinNew(c693, c268)
	var c726 *rlwe.Ciphertext
	_ = eval.Relinearize(c282, c726)
	c725, _ = eval.AddNew(c725, c726)
	c726, _ = eval.MulRelinNew(c663, c269)
	var c727 *rlwe.Ciphertext
	_ = eval.Relinearize(c726, c727)
	c725, _ = eval.AddNew(c725, c727)
	c9, _ = eval.MulRelinNew(c9, c37)
	_ = eval.Relinearize(c9, c727)
	var c107 *rlwe.Ciphertext
	c107, _ = eval.MulRelinNew(c10, c38)
	var c631 *rlwe.Ciphertext
	_ = eval.Relinearize(c107, c631)
	c727, _ = eval.AddNew(c727, c631)
	c631, _ = eval.MulRelinNew(c15, c39)
	var c632 *rlwe.Ciphertext
	_ = eval.Relinearize(c631, c632)
	c727, _ = eval.AddNew(c727, c632)
	c632, _ = eval.MulRelinNew(c16, c40)
	var c633 *rlwe.Ciphertext
	_ = eval.Relinearize(c632, c633)
	c727, _ = eval.AddNew(c727, c633)
	c727, _ = eval.MulRelinNew(c727, c727)
	_ = eval.Relinearize(c727, c633)
	var c286 *rlwe.Ciphertext
	c286, _ = eval.MulRelinNew(c633, c270)
	var c728 *rlwe.Ciphertext
	_ = eval.Relinearize(c286, c728)
	c725, _ = eval.AddNew(c725, c728)
	c725, _ = eval.MulRelinNew(c725, c725)
	_ = eval.Relinearize(c725, c728)
	var c421 *rlwe.Ciphertext
	c421, _ = eval.MulRelinNew(c728, c416)
	var c746 *rlwe.Ciphertext
	_ = eval.Relinearize(c421, c746)
	c745, _ = eval.AddNew(c745, c746)
	c678, _ = eval.MulRelinNew(c678, c267)
	_ = eval.Relinearize(c678, c746)
	var c309 *rlwe.Ciphertext
	c309, _ = eval.MulRelinNew(c663, c268)
	var c681 *rlwe.Ciphertext
	_ = eval.Relinearize(c309, c681)
	c746, _ = eval.AddNew(c746, c681)
	c13, _ = eval.MulRelinNew(c13, c37)
	_ = eval.Relinearize(c13, c681)
	var c134 *rlwe.Ciphertext
	c134, _ = eval.MulRelinNew(c14, c38)
	var c601 *rlwe.Ciphertext
	_ = eval.Relinearize(c134, c601)
	c681, _ = eval.AddNew(c681, c601)
	c601, _ = eval.MulRelinNew(c19, c39)
	var c602 *rlwe.Ciphertext
	_ = eval.Relinearize(c601, c602)
	c681, _ = eval.AddNew(c681, c602)
	c602, _ = eval.MulRelinNew(c20, c40)
	var c603 *rlwe.Ciphertext
	_ = eval.Relinearize(c602, c603)
	c681, _ = eval.AddNew(c681, c603)
	c681, _ = eval.MulRelinNew(c681, c681)
	_ = eval.Relinearize(c681, c603)
	var c311 *rlwe.Ciphertext
	c311, _ = eval.MulRelinNew(c603, c269)
	var c682 *rlwe.Ciphertext
	_ = eval.Relinearize(c311, c682)
	c746, _ = eval.AddNew(c746, c682)
	c14, _ = eval.MulRelinNew(c14, c37)
	_ = eval.Relinearize(c14, c682)
	var c143 *rlwe.Ciphertext
	c143, _ = eval.MulRelinNew(c15, c38)
	var c571 *rlwe.Ciphertext
	_ = eval.Relinearize(c143, c571)
	c682, _ = eval.AddNew(c682, c571)
	c571, _ = eval.MulRelinNew(c20, c39)
	var c572 *rlwe.Ciphertext
	_ = eval.Relinearize(c571, c572)
	c682, _ = eval.AddNew(c682, c572)
	c572, _ = eval.MulRelinNew(c21, c40)
	var c573 *rlwe.Ciphertext
	_ = eval.Relinearize(c572, c573)
	c682, _ = eval.AddNew(c682, c573)
	c682, _ = eval.MulRelinNew(c682, c682)
	_ = eval.Relinearize(c682, c573)
	var c313 *rlwe.Ciphertext
	c313, _ = eval.MulRelinNew(c573, c270)
	var c683 *rlwe.Ciphertext
	_ = eval.Relinearize(c313, c683)
	c746, _ = eval.AddNew(c746, c683)
	c746, _ = eval.MulRelinNew(c746, c746)
	_ = eval.Relinearize(c746, c683)
	var c423 *rlwe.Ciphertext
	c423, _ = eval.MulRelinNew(c683, c417)
	var c747 *rlwe.Ciphertext
	_ = eval.Relinearize(c423, c747)
	c745, _ = eval.AddNew(c745, c747)
	c663, _ = eval.MulRelinNew(c663, c267)
	_ = eval.Relinearize(c663, c747)
	var c318 *rlwe.Ciphertext
	c318, _ = eval.MulRelinNew(c633, c268)
	var c666 *rlwe.Ciphertext
	_ = eval.Relinearize(c318, c666)
	c747, _ = eval.AddNew(c747, c666)
	c666, _ = eval.MulRelinNew(c573, c269)
	var c667 *rlwe.Ciphertext
	_ = eval.Relinearize(c666, c667)
	c747, _ = eval.AddNew(c747, c667)
	c15, _ = eval.MulRelinNew(c15, c37)
	_ = eval.Relinearize(c15, c667)
	var c152 *rlwe.Ciphertext
	c152, _ = eval.MulRelinNew(c16, c38)
	var c501 *rlwe.Ciphertext
	_ = eval.Relinearize(c152, c501)
	c667, _ = eval.AddNew(c667, c501)
	c501, _ = eval.MulRelinNew(c21, c39)
	var c502 *rlwe.Ciphertext
	_ = eval.Relinearize(c501, c502)
	c667, _ = eval.AddNew(c667, c502)
	c502, _ = eval.MulRelinNew(c22, c40)
	var c503 *rlwe.Ciphertext
	_ = eval.Relinearize(c502, c503)
	c667, _ = eval.AddNew(c667, c503)
	c667, _ = eval.MulRelinNew(c667, c667)
	_ = eval.Relinearize(c667, c503)
	var c322 *rlwe.Ciphertext
	c322, _ = eval.MulRelinNew(c503, c270)
	var c668 *rlwe.Ciphertext
	_ = eval.Relinearize(c322, c668)
	c747, _ = eval.AddNew(c747, c668)
	c747, _ = eval.MulRelinNew(c747, c747)
	_ = eval.Relinearize(c747, c668)
	var c425 *rlwe.Ciphertext
	c425, _ = eval.MulRelinNew(c668, c418)
	var c748 *rlwe.Ciphertext
	_ = eval.Relinearize(c425, c748)
	c745, _ = eval.AddNew(c745, c748)
	c745, _ = eval.MulRelinNew(c745, c745)
	_ = eval.Relinearize(c745, c748)
	c693, _ = eval.MulRelinNew(c693, c267)
	var c695 *rlwe.Ciphertext
	_ = eval.Relinearize(c693, c695)
	c4, _ = eval.MulRelinNew(c4, c37)
	var c696 *rlwe.Ciphertext
	_ = eval.Relinearize(c4, c696)
	var c71 *rlwe.Ciphertext
	c71, _ = eval.MulRelinNew(c5, c38)
	var c697 *rlwe.Ciphertext
	_ = eval.Relinearize(c71, c697)
	c696, _ = eval.AddNew(c696, c697)
	c697, _ = eval.MulRelinNew(c10, c39)
	var c698 *rlwe.Ciphertext
	_ = eval.Relinearize(c697, c698)
	c696, _ = eval.AddNew(c696, c698)
	c698, _ = eval.MulRelinNew(c11, c40)
	var c699 *rlwe.Ciphertext
	_ = eval.Relinearize(c698, c699)
	c696, _ = eval.AddNew(c696, c699)
	c696, _ = eval.MulRelinNew(c696, c696)
	_ = eval.Relinearize(c696, c699)
	var c291 *rlwe.Ciphertext
	c291, _ = eval.MulRelinNew(c699, c268)
	var c701 *rlwe.Ciphertext
	_ = eval.Relinearize(c291, c701)
	c695, _ = eval.AddNew(c695, c701)
	c701, _ = eval.MulRelinNew(c633, c269)
	var c702 *rlwe.Ciphertext
	_ = eval.Relinearize(c701, c702)
	c695, _ = eval.AddNew(c695, c702)
	c10, _ = eval.MulRelinNew(c10, c37)
	_ = eval.Relinearize(c10, c702)
	var c116 *rlwe.Ciphertext
	c116, _ = eval.MulRelinNew(c11, c38)
	var c637 *rlwe.Ciphertext
	_ = eval.Relinearize(c116, c637)
	c702, _ = eval.AddNew(c702, c637)
	c637, _ = eval.MulRelinNew(c16, c39)
	var c638 *rlwe.Ciphertext
	_ = eval.Relinearize(c637, c638)
	c702, _ = eval.AddNew(c702, c638)
	c638, _ = eval.MulRelinNew(c17, c40)
	var c639 *rlwe.Ciphertext
	_ = eval.Relinearize(c638, c639)
	c702, _ = eval.AddNew(c702, c639)
	c702, _ = eval.MulRelinNew(c702, c702)
	_ = eval.Relinearize(c702, c639)
	var c295 *rlwe.Ciphertext
	c295, _ = eval.MulRelinNew(c639, c270)
	var c703 *rlwe.Ciphertext
	_ = eval.Relinearize(c295, c703)
	c695, _ = eval.AddNew(c695, c703)
	c695, _ = eval.MulRelinNew(c695, c695)
	_ = eval.Relinearize(c695, c703)
	var c437 *rlwe.Ciphertext
	c437, _ = eval.MulRelinNew(c703, c415)
	var c705 *rlwe.Ciphertext
	_ = eval.Relinearize(c437, c705)
	c699, _ = eval.MulRelinNew(c699, c267)
	var c706 *rlwe.Ciphertext
	_ = eval.Relinearize(c699, c706)
	c5, _ = eval.MulRelinNew(c5, c37)
	var c707 *rlwe.Ciphertext
	_ = eval.Relinearize(c5, c707)
	c6, _ = eval.MulRelinNew(c6, c38)
	var c708 *rlwe.Ciphertext
	_ = eval.Relinearize(c6, c708)
	c707, _ = eval.AddNew(c707, c708)
	c708, _ = eval.MulRelinNew(c11, c39)
	var c709 *rlwe.Ciphertext
	_ = eval.Relinearize(c708, c709)
	c707, _ = eval.AddNew(c707, c709)
	c709, _ = eval.MulRelinNew(c12, c40)
	var c710 *rlwe.Ciphertext
	_ = eval.Relinearize(c709, c710)
	c707, _ = eval.AddNew(c707, c710)
	c707, _ = eval.MulRelinNew(c707, c707)
	_ = eval.Relinearize(c707, c710)
	c710, _ = eval.MulRelinNew(c710, c268)
	var c712 *rlwe.Ciphertext
	_ = eval.Relinearize(c710, c712)
	c706, _ = eval.AddNew(c706, c712)
	c712, _ = eval.MulRelinNew(c639, c269)
	var c713 *rlwe.Ciphertext
	_ = eval.Relinearize(c712, c713)
	c706, _ = eval.AddNew(c706, c713)
	c11, _ = eval.MulRelinNew(c11, c37)
	_ = eval.Relinearize(c11, c713)
	c12, _ = eval.MulRelinNew(c12, c38)
	var c648 *rlwe.Ciphertext
	_ = eval.Relinearize(c12, c648)
	c713, _ = eval.AddNew(c713, c648)
	c648, _ = eval.MulRelinNew(c17, c39)
	var c649 *rlwe.Ciphertext
	_ = eval.Relinearize(c648, c649)
	c713, _ = eval.AddNew(c713, c649)
	c649, _ = eval.MulRelinNew(c18, c40)
	var c650 *rlwe.Ciphertext
	_ = eval.Relinearize(c649, c650)
	c713, _ = eval.AddNew(c713, c650)
	c713, _ = eval.MulRelinNew(c713, c713)
	_ = eval.Relinearize(c713, c650)
	var c304 *rlwe.Ciphertext
	c304, _ = eval.MulRelinNew(c650, c270)
	var c714 *rlwe.Ciphertext
	_ = eval.Relinearize(c304, c714)
	c706, _ = eval.AddNew(c706, c714)
	c706, _ = eval.MulRelinNew(c706, c706)
	_ = eval.Relinearize(c706, c714)
	c714, _ = eval.MulRelinNew(c714, c416)
	var c716 *rlwe.Ciphertext
	_ = eval.Relinearize(c714, c716)
	c705, _ = eval.AddNew(c705, c716)
	c633, _ = eval.MulRelinNew(c633, c267)
	_ = eval.Relinearize(c633, c716)
	var c327 *rlwe.Ciphertext
	c327, _ = eval.MulRelinNew(c639, c268)
	var c641 *rlwe.Ciphertext
	_ = eval.Relinearize(c327, c641)
	c716, _ = eval.AddNew(c716, c641)
	c641, _ = eval.MulRelinNew(c503, c269)
	var c642 *rlwe.Ciphertext
	_ = eval.Relinearize(c641, c642)
	c716, _ = eval.AddNew(c716, c642)
	c16, _ = eval.MulRelinNew(c16, c37)
	_ = eval.Relinearize(c16, c642)
	var c161 *rlwe.Ciphertext
	c161, _ = eval.MulRelinNew(c17, c38)
	var c507 *rlwe.Ciphertext
	_ = eval.Relinearize(c161, c507)
	c642, _ = eval.AddNew(c642, c507)
	c507, _ = eval.MulRelinNew(c22, c39)
	var c508 *rlwe.Ciphertext
	_ = eval.Relinearize(c507, c508)
	c642, _ = eval.AddNew(c642, c508)
	c508, _ = eval.MulRelinNew(c23, c40)
	var c509 *rlwe.Ciphertext
	_ = eval.Relinearize(c508, c509)
	c642, _ = eval.AddNew(c642, c509)
	c642, _ = eval.MulRelinNew(c642, c642)
	_ = eval.Relinearize(c642, c509)
	var c331 *rlwe.Ciphertext
	c331, _ = eval.MulRelinNew(c509, c270)
	var c643 *rlwe.Ciphertext
	_ = eval.Relinearize(c331, c643)
	c716, _ = eval.AddNew(c716, c643)
	c716, _ = eval.MulRelinNew(c716, c716)
	_ = eval.Relinearize(c716, c643)
	var c441 *rlwe.Ciphertext
	c441, _ = eval.MulRelinNew(c643, c417)
	var c717 *rlwe.Ciphertext
	_ = eval.Relinearize(c441, c717)
	c705, _ = eval.AddNew(c705, c717)
	c639, _ = eval.MulRelinNew(c639, c267)
	_ = eval.Relinearize(c639, c717)
	c650, _ = eval.MulRelinNew(c650, c268)
	var c652 *rlwe.Ciphertext
	_ = eval.Relinearize(c650, c652)
	c717, _ = eval.AddNew(c717, c652)
	c652, _ = eval.MulRelinNew(c509, c269)
	var c653 *rlwe.Ciphertext
	_ = eval.Relinearize(c652, c653)
	c717, _ = eval.AddNew(c717, c653)
	c17, _ = eval.MulRelinNew(c17, c37)
	_ = eval.Relinearize(c17, c653)
	c18, _ = eval.MulRelinNew(c18, c38)
	var c528 *rlwe.Ciphertext
	_ = eval.Relinearize(c18, c528)
	c653, _ = eval.AddNew(c653, c528)
	c528, _ = eval.MulRelinNew(c23, c39)
	var c529 *rlwe.Ciphertext
	_ = eval.Relinearize(c528, c529)
	c653, _ = eval.AddNew(c653, c529)
	c529, _ = eval.MulRelinNew(c24, c40)
	var c530 *rlwe.Ciphertext
	_ = eval.Relinearize(c529, c530)
	c653, _ = eval.AddNew(c653, c530)
	c653, _ = eval.MulRelinNew(c653, c653)
	_ = eval.Relinearize(c653, c530)
	var c340 *rlwe.Ciphertext
	c340, _ = eval.MulRelinNew(c530, c270)
	var c654 *rlwe.Ciphertext
	_ = eval.Relinearize(c340, c654)
	c717, _ = eval.AddNew(c717, c654)
	c717, _ = eval.MulRelinNew(c717, c717)
	_ = eval.Relinearize(c717, c654)
	var c443 *rlwe.Ciphertext
	c443, _ = eval.MulRelinNew(c654, c418)
	var c718 *rlwe.Ciphertext
	_ = eval.Relinearize(c443, c718)
	c705, _ = eval.AddNew(c705, c718)
	c705, _ = eval.MulRelinNew(c705, c705)
	_ = eval.Relinearize(c705, c718)
	c603, _ = eval.MulRelinNew(c603, c267)
	var c605 *rlwe.Ciphertext
	_ = eval.Relinearize(c603, c605)
	var c345 *rlwe.Ciphertext
	c345, _ = eval.MulRelinNew(c573, c268)
	var c606 *rlwe.Ciphertext
	_ = eval.Relinearize(c345, c606)
	c605, _ = eval.AddNew(c605, c606)
	c19, _ = eval.MulRelinNew(c19, c37)
	_ = eval.Relinearize(c19, c606)
	var c179 *rlwe.Ciphertext
	c179, _ = eval.MulRelinNew(c20, c38)
	var c608 *rlwe.Ciphertext
	_ = eval.Relinearize(c179, c608)
	c606, _ = eval.AddNew(c606, c608)
	c608, _ = eval.MulRelinNew(c25, c39)
	var c609 *rlwe.Ciphertext
	_ = eval.Relinearize(c608, c609)
	c606, _ = eval.AddNew(c606, c609)
	c609, _ = eval.MulRelinNew(c26, c40)
	var c610 *rlwe.Ciphertext
	_ = eval.Relinearize(c609, c610)
	c606, _ = eval.AddNew(c606, c610)
	c606, _ = eval.MulRelinNew(c606, c606)
	_ = eval.Relinearize(c606, c610)
	var c347 *rlwe.Ciphertext
	c347, _ = eval.MulRelinNew(c610, c269)
	var c612 *rlwe.Ciphertext
	_ = eval.Relinearize(c347, c612)
	c605, _ = eval.AddNew(c605, c612)
	c20, _ = eval.MulRelinNew(c20, c37)
	_ = eval.Relinearize(c20, c612)
	var c188 *rlwe.Ciphertext
	c188, _ = eval.MulRelinNew(c21, c38)
	var c578 *rlwe.Ciphertext
	_ = eval.Relinearize(c188, c578)
	c612, _ = eval.AddNew(c612, c578)
	c578, _ = eval.MulRelinNew(c26, c39)
	var c579 *rlwe.Ciphertext
	_ = eval.Relinearize(c578, c579)
	c612, _ = eval.AddNew(c612, c579)
	c579, _ = eval.MulRelinNew(c27, c40)
	var c580 *rlwe.Ciphertext
	_ = eval.Relinearize(c579, c580)
	c612, _ = eval.AddNew(c612, c580)
	c612, _ = eval.MulRelinNew(c612, c612)
	_ = eval.Relinearize(c612, c580)
	var c349 *rlwe.Ciphertext
	c349, _ = eval.MulRelinNew(c580, c270)
	var c613 *rlwe.Ciphertext
	_ = eval.Relinearize(c349, c613)
	c605, _ = eval.AddNew(c605, c613)
	c605, _ = eval.MulRelinNew(c605, c605)
	_ = eval.Relinearize(c605, c613)
	var c473 *rlwe.Ciphertext
	c473, _ = eval.MulRelinNew(c613, c415)
	var c615 *rlwe.Ciphertext
	_ = eval.Relinearize(c473, c615)
	c573, _ = eval.MulRelinNew(c573, c267)
	var c575 *rlwe.Ciphertext
	_ = eval.Relinearize(c573, c575)
	var c354 *rlwe.Ciphertext
	c354, _ = eval.MulRelinNew(c503, c268)
	var c576 *rlwe.Ciphertext
	_ = eval.Relinearize(c354, c576)
	c575, _ = eval.AddNew(c575, c576)
	c576, _ = eval.MulRelinNew(c580, c269)
	var c582 *rlwe.Ciphertext
	_ = eval.Relinearize(c576, c582)
	c575, _ = eval.AddNew(c575, c582)
	c21, _ = eval.MulRelinNew(c21, c37)
	_ = eval.Relinearize(c21, c582)
	var c197 *rlwe.Ciphertext
	c197, _ = eval.MulRelinNew(c22, c38)
	var c513 *rlwe.Ciphertext
	_ = eval.Relinearize(c197, c513)
	c582, _ = eval.AddNew(c582, c513)
	c513, _ = eval.MulRelinNew(c27, c39)
	var c514 *rlwe.Ciphertext
	_ = eval.Relinearize(c513, c514)
	c582, _ = eval.AddNew(c582, c514)
	c514, _ = eval.MulRelinNew(c28, c40)
	var c515 *rlwe.Ciphertext
	_ = eval.Relinearize(c514, c515)
	c582, _ = eval.AddNew(c582, c515)
	c582, _ = eval.MulRelinNew(c582, c582)
	_ = eval.Relinearize(c582, c515)
	var c358 *rlwe.Ciphertext
	c358, _ = eval.MulRelinNew(c515, c270)
	var c583 *rlwe.Ciphertext
	_ = eval.Relinearize(c358, c583)
	c575, _ = eval.AddNew(c575, c583)
	c575, _ = eval.MulRelinNew(c575, c575)
	_ = eval.Relinearize(c575, c583)
	var c475 *rlwe.Ciphertext
	c475, _ = eval.MulRelinNew(c583, c416)
	var c616 *rlwe.Ciphertext
	_ = eval.Relinearize(c475, c616)
	c615, _ = eval.AddNew(c615, c616)
	c610, _ = eval.MulRelinNew(c610, c267)
	_ = eval.Relinearize(c610, c616)
	var c381 *rlwe.Ciphertext
	c381, _ = eval.MulRelinNew(c580, c268)
	var c618 *rlwe.Ciphertext
	_ = eval.Relinearize(c381, c618)
	c616, _ = eval.AddNew(c616, c618)
	c25, _ = eval.MulRelinNew(c25, c37)
	_ = eval.Relinearize(c25, c618)
	var c224 *rlwe.Ciphertext
	c224, _ = eval.MulRelinNew(c26, c38)
	var c620 *rlwe.Ciphertext
	_ = eval.Relinearize(c224, c620)
	c618, _ = eval.AddNew(c618, c620)
	c31, _ = eval.MulRelinNew(c31, c39)
	_ = eval.Relinearize(c31, c620)
	c618, _ = eval.AddNew(c618, c620)
	c620, _ = eval.MulRelinNew(c32, c40)
	var c622 *rlwe.Ciphertext
	_ = eval.Relinearize(c620, c622)
	c618, _ = eval.AddNew(c618, c622)
	c618, _ = eval.MulRelinNew(c618, c618)
	_ = eval.Relinearize(c618, c622)
	c622, _ = eval.MulRelinNew(c622, c269)
	var c624 *rlwe.Ciphertext
	_ = eval.Relinearize(c622, c624)
	c616, _ = eval.AddNew(c616, c624)
	c26, _ = eval.MulRelinNew(c26, c37)
	_ = eval.Relinearize(c26, c624)
	var c233 *rlwe.Ciphertext
	c233, _ = eval.MulRelinNew(c27, c38)
	var c590 *rlwe.Ciphertext
	_ = eval.Relinearize(c233, c590)
	c624, _ = eval.AddNew(c624, c590)
	c32, _ = eval.MulRelinNew(c32, c39)
	_ = eval.Relinearize(c32, c590)
	c624, _ = eval.AddNew(c624, c590)
	c590, _ = eval.MulRelinNew(c33, c40)
	var c592 *rlwe.Ciphertext
	_ = eval.Relinearize(c590, c592)
	c624, _ = eval.AddNew(c624, c592)
	c624, _ = eval.MulRelinNew(c624, c624)
	_ = eval.Relinearize(c624, c592)
	var c385 *rlwe.Ciphertext
	c385, _ = eval.MulRelinNew(c592, c270)
	var c625 *rlwe.Ciphertext
	_ = eval.Relinearize(c385, c625)
	c616, _ = eval.AddNew(c616, c625)
	c616, _ = eval.MulRelinNew(c616, c616)
	_ = eval.Relinearize(c616, c625)
	c625, _ = eval.MulRelinNew(c625, c417)
	var c627 *rlwe.Ciphertext
	_ = eval.Relinearize(c625, c627)
	c615, _ = eval.AddNew(c615, c627)
	c580, _ = eval.MulRelinNew(c580, c267)
	_ = eval.Relinearize(c580, c627)
	var c390 *rlwe.Ciphertext
	c390, _ = eval.MulRelinNew(c515, c268)
	var c588 *rlwe.Ciphertext
	_ = eval.Relinearize(c390, c588)
	c627, _ = eval.AddNew(c627, c588)
	c592, _ = eval.MulRelinNew(c592, c269)
	_ = eval.Relinearize(c592, c588)
	c627, _ = eval.AddNew(c627, c588)
	c27, _ = eval.MulRelinNew(c27, c37)
	_ = eval.Relinearize(c27, c588)
	var c242 *rlwe.Ciphertext
	c242, _ = eval.MulRelinNew(c28, c38)
	var c545 *rlwe.Ciphertext
	_ = eval.Relinearize(c242, c545)
	c588, _ = eval.AddNew(c588, c545)
	c33, _ = eval.MulRelinNew(c33, c39)
	_ = eval.Relinearize(c33, c545)
	c588, _ = eval.AddNew(c588, c545)
	c545, _ = eval.MulRelinNew(c34, c40)
	var c547 *rlwe.Ciphertext
	_ = eval.Relinearize(c545, c547)
	c588, _ = eval.AddNew(c588, c547)
	c588, _ = eval.MulRelinNew(c588, c588)
	_ = eval.Relinearize(c588, c547)
	var c394 *rlwe.Ciphertext
	c394, _ = eval.MulRelinNew(c547, c270)
	var c595 *rlwe.Ciphertext
	_ = eval.Relinearize(c394, c595)
	c627, _ = eval.AddNew(c627, c595)
	c627, _ = eval.MulRelinNew(c627, c627)
	_ = eval.Relinearize(c627, c595)
	var c479 *rlwe.Ciphertext
	c479, _ = eval.MulRelinNew(c595, c418)
	var c628 *rlwe.Ciphertext
	_ = eval.Relinearize(c479, c628)
	c615, _ = eval.AddNew(c615, c628)
	c615, _ = eval.MulRelinNew(c615, c615)
	_ = eval.Relinearize(c615, c628)
	c728, _ = eval.MulRelinNew(c728, c415)
	var c730 *rlwe.Ciphertext
	_ = eval.Relinearize(c728, c730)
	c703, _ = eval.MulRelinNew(c703, c416)
	var c731 *rlwe.Ciphertext
	_ = eval.Relinearize(c703, c731)
	c730, _ = eval.AddNew(c730, c731)
	c731, _ = eval.MulRelinNew(c668, c417)
	var c732 *rlwe.Ciphertext
	_ = eval.Relinearize(c731, c732)
	c730, _ = eval.AddNew(c730, c732)
	c732, _ = eval.MulRelinNew(c643, c418)
	var c733 *rlwe.Ciphertext
	_ = eval.Relinearize(c732, c733)
	c730, _ = eval.AddNew(c730, c733)
	c730, _ = eval.MulRelinNew(c730, c730)
	_ = eval.Relinearize(c730, c733)
	c503, _ = eval.MulRelinNew(c503, c267)
	var c505 *rlwe.Ciphertext
	_ = eval.Relinearize(c503, c505)
	var c363 *rlwe.Ciphertext
	c363, _ = eval.MulRelinNew(c509, c268)
	var c511 *rlwe.Ciphertext
	_ = eval.Relinearize(c363, c511)
	c505, _ = eval.AddNew(c505, c511)
	c511, _ = eval.MulRelinNew(c515, c269)
	var c517 *rlwe.Ciphertext
	_ = eval.Relinearize(c511, c517)
	c505, _ = eval.AddNew(c505, c517)
	c22, _ = eval.MulRelinNew(c22, c37)
	_ = eval.Relinearize(c22, c517)
	var c206 *rlwe.Ciphertext
	c206, _ = eval.MulRelinNew(c23, c38)
	var c519 *rlwe.Ciphertext
	_ = eval.Relinearize(c206, c519)
	c517, _ = eval.AddNew(c517, c519)
	c519, _ = eval.MulRelinNew(c28, c39)
	var c520 *rlwe.Ciphertext
	_ = eval.Relinearize(c519, c520)
	c517, _ = eval.AddNew(c517, c520)
	c520, _ = eval.MulRelinNew(c29, c40)
	var c521 *rlwe.Ciphertext
	_ = eval.Relinearize(c520, c521)
	c517, _ = eval.AddNew(c517, c521)
	c517, _ = eval.MulRelinNew(c517, c517)
	_ = eval.Relinearize(c517, c521)
	var c367 *rlwe.Ciphertext
	c367, _ = eval.MulRelinNew(c521, c270)
	var c523 *rlwe.Ciphertext
	_ = eval.Relinearize(c367, c523)
	c505, _ = eval.AddNew(c505, c523)
	c505, _ = eval.MulRelinNew(c505, c505)
	_ = eval.Relinearize(c505, c523)
	var c491 *rlwe.Ciphertext
	c491, _ = eval.MulRelinNew(c523, c415)
	var c525 *rlwe.Ciphertext
	_ = eval.Relinearize(c491, c525)
	c509, _ = eval.MulRelinNew(c509, c267)
	var c526 *rlwe.Ciphertext
	_ = eval.Relinearize(c509, c526)
	c530, _ = eval.MulRelinNew(c530, c268)
	var c532 *rlwe.Ciphertext
	_ = eval.Relinearize(c530, c532)
	c526, _ = eval.AddNew(c526, c532)
	c532, _ = eval.MulRelinNew(c521, c269)
	var c533 *rlwe.Ciphertext
	_ = eval.Relinearize(c532, c533)
	c526, _ = eval.AddNew(c526, c533)
	c23, _ = eval.MulRelinNew(c23, c37)
	_ = eval.Relinearize(c23, c533)
	c24, _ = eval.MulRelinNew(c24, c38)
	var c535 *rlwe.Ciphertext
	_ = eval.Relinearize(c24, c535)
	c533, _ = eval.AddNew(c533, c535)
	c535, _ = eval.MulRelinNew(c29, c39)
	var c536 *rlwe.Ciphertext
	_ = eval.Relinearize(c535, c536)
	c533, _ = eval.AddNew(c533, c536)
	c536, _ = eval.MulRelinNew(c30, c40)
	var c537 *rlwe.Ciphertext
	_ = eval.Relinearize(c536, c537)
	c533, _ = eval.AddNew(c533, c537)
	c533, _ = eval.MulRelinNew(c533, c533)
	_ = eval.Relinearize(c533, c537)
	var c376 *rlwe.Ciphertext
	c376, _ = eval.MulRelinNew(c537, c270)
	var c539 *rlwe.Ciphertext
	_ = eval.Relinearize(c376, c539)
	c526, _ = eval.AddNew(c526, c539)
	c526, _ = eval.MulRelinNew(c526, c526)
	_ = eval.Relinearize(c526, c539)
	var c493 *rlwe.Ciphertext
	c493, _ = eval.MulRelinNew(c539, c416)
	var c541 *rlwe.Ciphertext
	_ = eval.Relinearize(c493, c541)
	c525, _ = eval.AddNew(c525, c541)
	c515, _ = eval.MulRelinNew(c515, c267)
	_ = eval.Relinearize(c515, c541)
	var c399 *rlwe.Ciphertext
	c399, _ = eval.MulRelinNew(c521, c268)
	var c543 *rlwe.Ciphertext
	_ = eval.Relinearize(c399, c543)
	c541, _ = eval.AddNew(c541, c543)
	c547, _ = eval.MulRelinNew(c547, c269)
	_ = eval.Relinearize(c547, c543)
	c541, _ = eval.AddNew(c541, c543)
	c28, _ = eval.MulRelinNew(c28, c37)
	_ = eval.Relinearize(c28, c543)
	var c251 *rlwe.Ciphertext
	c251, _ = eval.MulRelinNew(c29, c38)
	var c551 *rlwe.Ciphertext
	_ = eval.Relinearize(c251, c551)
	c543, _ = eval.AddNew(c543, c551)
	c34, _ = eval.MulRelinNew(c34, c39)
	_ = eval.Relinearize(c34, c551)
	c543, _ = eval.AddNew(c543, c551)
	c551, _ = eval.MulRelinNew(c35, c40)
	var c553 *rlwe.Ciphertext
	_ = eval.Relinearize(c551, c553)
	c543, _ = eval.AddNew(c543, c553)
	c543, _ = eval.MulRelinNew(c543, c543)
	_ = eval.Relinearize(c543, c553)
	var c403 *rlwe.Ciphertext
	c403, _ = eval.MulRelinNew(c553, c270)
	var c555 *rlwe.Ciphertext
	_ = eval.Relinearize(c403, c555)
	c541, _ = eval.AddNew(c541, c555)
	c541, _ = eval.MulRelinNew(c541, c541)
	_ = eval.Relinearize(c541, c555)
	var c495 *rlwe.Ciphertext
	c495, _ = eval.MulRelinNew(c555, c417)
	var c557 *rlwe.Ciphertext
	_ = eval.Relinearize(c495, c557)
	c525, _ = eval.AddNew(c525, c557)
	c521, _ = eval.MulRelinNew(c521, c267)
	_ = eval.Relinearize(c521, c557)
	c537, _ = eval.MulRelinNew(c537, c268)
	_ = eval.Relinearize(c537, c267)
	c557, _ = eval.AddNew(c557, c267)
	c553, _ = eval.MulRelinNew(c553, c269)
	_ = eval.Relinearize(c553, c267)
	c557, _ = eval.AddNew(c557, c267)
	c29, _ = eval.MulRelinNew(c29, c37)
	_ = eval.Relinearize(c29, c267)
	c30, _ = eval.MulRelinNew(c30, c38)
	_ = eval.Relinearize(c30, c268)
	c267, _ = eval.AddNew(c267, c268)
	c35, _ = eval.MulRelinNew(c35, c39)
	_ = eval.Relinearize(c35, c268)
	c267, _ = eval.AddNew(c267, c268)
	c36, _ = eval.MulRelinNew(c36, c40)
	_ = eval.Relinearize(c36, c268)
	c267, _ = eval.AddNew(c267, c268)
	c267, _ = eval.MulRelinNew(c267, c267)
	_ = eval.Relinearize(c267, c268)
	c268, _ = eval.MulRelinNew(c268, c270)
	_ = eval.Relinearize(c268, c40)
	c557, _ = eval.AddNew(c557, c40)
	c557, _ = eval.MulRelinNew(c557, c557)
	_ = eval.Relinearize(c557, c40)
	c40, _ = eval.MulRelinNew(c40, c418)
	_ = eval.Relinearize(c40, c270)
	c525, _ = eval.AddNew(c525, c270)
	c525, _ = eval.MulRelinNew(c525, c525)
	_ = eval.Relinearize(c525, c270)
	c269, _ = eval.MulRelinNew(c643, c415)
	_ = eval.Relinearize(c269, c37)
	c654, _ = eval.MulRelinNew(c654, c416)
	_ = eval.Relinearize(c654, c39)
	c37, _ = eval.AddNew(c37, c39)
	c39, _ = eval.MulRelinNew(c523, c417)
	_ = eval.Relinearize(c39, c38)
	c37, _ = eval.AddNew(c37, c38)
	c539, _ = eval.MulRelinNew(c539, c418)
	_ = eval.Relinearize(c539, c38)
	c37, _ = eval.AddNew(c37, c38)
	c37, _ = eval.MulRelinNew(c37, c37)
	_ = eval.Relinearize(c37, c38)
	var c455 *rlwe.Ciphertext
	c455, _ = eval.MulRelinNew(c668, c415)
	var c670 *rlwe.Ciphertext
	_ = eval.Relinearize(c455, c670)
	c643, _ = eval.MulRelinNew(c643, c416)
	var c671 *rlwe.Ciphertext
	_ = eval.Relinearize(c643, c671)
	c670, _ = eval.AddNew(c670, c671)
	c671, _ = eval.MulRelinNew(c583, c417)
	var c672 *rlwe.Ciphertext
	_ = eval.Relinearize(c671, c672)
	c670, _ = eval.AddNew(c670, c672)
	c672, _ = eval.MulRelinNew(c523, c418)
	var c673 *rlwe.Ciphertext
	_ = eval.Relinearize(c672, c673)
	c670, _ = eval.AddNew(c670, c673)
	c670, _ = eval.MulRelinNew(c670, c670)
	_ = eval.Relinearize(c670, c673)
	var c482 *rlwe.Ciphertext
	c482, _ = eval.MulRelinNew(c583, c415)
	var c585 *rlwe.Ciphertext
	_ = eval.Relinearize(c482, c585)
	c523, _ = eval.MulRelinNew(c523, c416)
	var c586 *rlwe.Ciphertext
	_ = eval.Relinearize(c523, c586)
	c585, _ = eval.AddNew(c585, c586)
	c595, _ = eval.MulRelinNew(c595, c417)
	_ = eval.Relinearize(c595, c586)
	c585, _ = eval.AddNew(c585, c586)
	c555, _ = eval.MulRelinNew(c555, c418)
	_ = eval.Relinearize(c555, c586)
	c585, _ = eval.AddNew(c585, c586)
	c585, _ = eval.MulRelinNew(c585, c585)
	_ = eval.Relinearize(c585, c586)
	c683, _ = eval.MulRelinNew(c683, c415)
	_ = eval.Relinearize(c683, c415)
	c668, _ = eval.MulRelinNew(c668, c416)
	_ = eval.Relinearize(c668, c416)
	c415, _ = eval.AddNew(c415, c416)
	c613, _ = eval.MulRelinNew(c613, c417)
	_ = eval.Relinearize(c613, c416)
	c415, _ = eval.AddNew(c415, c416)
	c583, _ = eval.MulRelinNew(c583, c418)
	_ = eval.Relinearize(c583, c416)
	c415, _ = eval.AddNew(c415, c416)
	c415, _ = eval.MulRelinNew(c415, c415)
	_ = eval.Relinearize(c415, c416)

	// Store outputs
	encryptedOutputs["out_0_0"] = c748
	encryptedOutputs["out_0_2"] = c718
	encryptedOutputs["out_2_0"] = c628
	encryptedOutputs["out_0_1"] = c733
	encryptedOutputs["out_2_2"] = c270
	encryptedOutputs["out_1_2"] = c38
	encryptedOutputs["out_1_1"] = c673
	encryptedOutputs["out_2_1"] = c586
	encryptedOutputs["out_1_0"] = c416
}


func main() {
	// CKKS Parameters (generated from CKKSParamSelector)
	// LogN=14 (n=16384, slots=8192)
	// MaxLevel=7, LogScale=40
	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            14,
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{45, 45},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois keys for rotations
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}
	gks := kgen.GenGaloisKeysNew(galoisElements, sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Encoder, Encryptor, Decryptor, Evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	// Input/Output maps
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)

	// TODO: Prepare your inputs here
	// Example:
	// values := make([]float64, params.MaxSlots())
	// for i := range values { values[i] = float64(i) }
	// pt := hefloat.NewPlaintext(params, params.MaxLevel())
	// encoder.Encode(values, pt)
	// ct, _ := enc.EncryptNew(pt)
	// encryptedInputs["c0"] = ct

	// Run computation
	deep_conv(encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)

	// Decrypt and print results
	for name, ct := range encryptedOutputs {
		pt := dec.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		encoder.Decode(pt, values)
		fmt.Printf("%s: [%.4f, %.4f, %.4f, ...]\n", name, values[0], values[1], values[2])
	}

	_ = encodedOutputs
	fmt.Println("CKKS computation completed!")
}
