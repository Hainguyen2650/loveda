# Blue Channel Quantization Check

This report compares `R`, `G`, and `B` channel histograms using simple quantization-oriented diagnostics:
- histogram roughness via first and second differences,
- even-vs-odd parity bias,
- residue bias under `mod 2`, `mod 4`, and `mod 8`,
- and average unique intensity count per image.

Interpretation rule:
- If `B` consistently has higher roughness and stronger modulo residue bias than `R/G`, that supports the quantization/discretization hypothesis.

## Train Urban

- `R`: roughness-1=0.0002494472, roughness-2=0.0001568362, parity-bias=0.0330225527, midrange peak/trough=2.535929, mod8 dominant residue=0, mod8 uniformity-l1=0.0578922927
- `G`: roughness-1=0.0002737784, roughness-2=0.0001567279, parity-bias=0.0328831375, midrange peak/trough=8.287184, mod8 dominant residue=0, mod8 uniformity-l1=0.058223784
- `B`: roughness-1=0.0008615466, roughness-2=0.001454946, parity-bias=0.0329086483, midrange peak/trough=20.31831, mod8 dominant residue=3, mod8 uniformity-l1=0.0804697871

## Train Rural

- `R`: roughness-1=0.0001986651, roughness-2=0.0001177914, parity-bias=0.0243866146, midrange peak/trough=2.318834, mod8 dominant residue=0, mod8 uniformity-l1=0.0433353186
- `G`: roughness-1=0.0002099007, roughness-2=0.0001127377, parity-bias=0.0239222646, midrange peak/trough=3.006431, mod8 dominant residue=0, mod8 uniformity-l1=0.0422575176
- `B`: roughness-1=0.0004667106, roughness-2=0.0006894015, parity-bias=0.0240049064, midrange peak/trough=3.674332, mod8 dominant residue=0, mod8 uniformity-l1=0.041805774

## Val Urban

- `R`: roughness-1=0.0002880036, roughness-2=0.0001799985, parity-bias=0.0390917361, midrange peak/trough=2.983961, mod8 dominant residue=0, mod8 uniformity-l1=0.0677992404
- `G`: roughness-1=0.0003086861, roughness-2=0.000174251, parity-bias=0.038811028, midrange peak/trough=5.636628, mod8 dominant residue=0, mod8 uniformity-l1=0.0681032836
- `B`: roughness-1=0.0008858882, roughness-2=0.0014655463, parity-bias=0.0386765003, midrange peak/trough=13.854682, mod8 dominant residue=2, mod8 uniformity-l1=0.0931597054

## Val Rural

- `R`: roughness-1=0.0002193863, roughness-2=0.0001420638, parity-bias=0.0291130841, midrange peak/trough=3.438237, mod8 dominant residue=0, mod8 uniformity-l1=0.0507740974
- `G`: roughness-1=0.0002253059, roughness-2=0.0001338855, parity-bias=0.0281307995, midrange peak/trough=2.250702, mod8 dominant residue=0, mod8 uniformity-l1=0.0493714809
- `B`: roughness-1=0.000407835, roughness-2=0.0005420542, parity-bias=0.0282997787, midrange peak/trough=5.371247, mod8 dominant residue=0, mod8 uniformity-l1=0.0472087264

## Test Urban

- `R`: roughness-1=0.0003450624, roughness-2=0.0002076823, parity-bias=0.0370237231, midrange peak/trough=5.50916, mod8 dominant residue=0, mod8 uniformity-l1=0.0665320754
- `G`: roughness-1=0.0003387134, roughness-2=0.000194781, parity-bias=0.0362274051, midrange peak/trough=12.087074, mod8 dominant residue=0, mod8 uniformity-l1=0.0623220801
- `B`: roughness-1=0.0008090194, roughness-2=0.0012767861, parity-bias=0.0363504887, midrange peak/trough=15.376204, mod8 dominant residue=2, mod8 uniformity-l1=0.0878240764

## Test Rural

- `R`: roughness-1=0.0002458682, roughness-2=0.0001382088, parity-bias=0.0272045135, midrange peak/trough=3.055536, mod8 dominant residue=0, mod8 uniformity-l1=0.0485837758
- `G`: roughness-1=0.000259379, roughness-2=0.0001348759, parity-bias=0.0270988345, midrange peak/trough=4.431204, mod8 dominant residue=0, mod8 uniformity-l1=0.0489163995
- `B`: roughness-1=0.0007551522, roughness-2=0.0012509287, parity-bias=0.027094841, midrange peak/trough=11.41185, mod8 dominant residue=3, mod8 uniformity-l1=0.0723196864

Recommendation:
- Treat the blue-channel histogram as a dataset property if its roughness and modulo bias remain systematically stronger after reruns.
- Keep normalization explicit and use moderate color augmentation if the blue channel remains more discretized than red/green.
