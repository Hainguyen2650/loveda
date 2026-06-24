# Blue Channel Quantization Check

This report compares `R`, `G`, and `B` channel histograms using simple quantization-oriented diagnostics:
- histogram roughness via first and second differences,
- even-vs-odd parity bias,
- residue bias under `mod 2`, `mod 4`, and `mod 8`,
- and average unique intensity count per image.

Interpretation rule:
- If `B` consistently has higher roughness and stronger modulo residue bias than `R/G`, that supports the quantization/discretization hypothesis.

## Train Urban

- `R`: roughness-1=0.0003111207, roughness-2=0.0002025902, parity-bias=0.0478355282, midrange peak/trough=2.857847, mod8 dominant residue=0, mod8 uniformity-l1=0.0844135218
- `G`: roughness-1=0.0003326508, roughness-2=0.0002052901, parity-bias=0.0480124480, midrange peak/trough=6.892083, mod8 dominant residue=0, mod8 uniformity-l1=0.0849403576
- `B`: roughness-1=0.0009645243, roughness-2=0.0016060211, parity-bias=0.0479205075, midrange peak/trough=16.376532, mod8 dominant residue=3, mod8 uniformity-l1=0.1008323732

## Train Rural

- `R`: roughness-1=0.0002169585, roughness-2=0.0001240631, parity-bias=0.0290760736, midrange peak/trough=2.252078, mod8 dominant residue=0, mod8 uniformity-l1=0.0516199221
- `G`: roughness-1=0.0002315602, roughness-2=0.0001225256, parity-bias=0.0285838101, midrange peak/trough=3.144890, mod8 dominant residue=0, mod8 uniformity-l1=0.0506717687
- `B`: roughness-1=0.0005119234, roughness-2=0.0007628838, parity-bias=0.0285551454, midrange peak/trough=3.990345, mod8 dominant residue=0, mod8 uniformity-l1=0.0411462686

## Val Urban

- `R`: roughness-1=0.0002862522, roughness-2=0.0001668796, parity-bias=0.0386317745, midrange peak/trough=3.086253, mod8 dominant residue=0, mod8 uniformity-l1=0.0677654514
- `G`: roughness-1=0.0003076276, roughness-2=0.0001667113, parity-bias=0.0385148289, midrange peak/trough=4.799124, mod8 dominant residue=0, mod8 uniformity-l1=0.0677176973
- `B`: roughness-1=0.0008850643, roughness-2=0.0014810208, parity-bias=0.0383343605, midrange peak/trough=10.868020, mod8 dominant residue=3, mod8 uniformity-l1=0.0854309257

## Val Rural

- `R`: roughness-1=0.0001777197, roughness-2=0.0000838738, parity-bias=0.0183252981, midrange peak/trough=3.364602, mod8 dominant residue=0, mod8 uniformity-l1=0.0323293171
- `G`: roughness-1=0.0001856240, roughness-2=0.0000801772, parity-bias=0.0177245467, midrange peak/trough=2.327630, mod8 dominant residue=0, mod8 uniformity-l1=0.0312076230
- `B`: roughness-1=0.0003799551, roughness-2=0.0005371288, parity-bias=0.0180429605, midrange peak/trough=5.063911, mod8 dominant residue=3, mod8 uniformity-l1=0.0347274523

## Test Urban

- `R`: roughness-1=0.0003016800, roughness-2=0.0001593839, parity-bias=0.0322343757, midrange peak/trough=5.182791, mod8 dominant residue=0, mod8 uniformity-l1=0.0578611839
- `G`: roughness-1=0.0002923491, roughness-2=0.0001577040, parity-bias=0.0321230353, midrange peak/trough=9.599550, mod8 dominant residue=0, mod8 uniformity-l1=0.0560592651
- `B`: roughness-1=0.0008472311, roughness-2=0.0014019772, parity-bias=0.0323017934, midrange peak/trough=10.234141, mod8 dominant residue=2, mod8 uniformity-l1=0.0846841882

## Test Rural

- `R`: roughness-1=0.0002183484, roughness-2=0.0000951337, parity-bias=0.0197411170, midrange peak/trough=2.985329, mod8 dominant residue=0, mod8 uniformity-l1=0.0359033405
- `G`: roughness-1=0.0002269081, roughness-2=0.0000952762, parity-bias=0.0198047650, midrange peak/trough=4.418422, mod8 dominant residue=0, mod8 uniformity-l1=0.0356745642
- `B`: roughness-1=0.0007566158, roughness-2=0.0012799493, parity-bias=0.0196886610, midrange peak/trough=10.778582, mod8 dominant residue=3, mod8 uniformity-l1=0.0884417061

Recommendation:
- Treat the blue-channel histogram as a dataset property if its roughness and modulo bias remain systematically stronger after reruns.
- Keep normalization explicit and use moderate color augmentation if the blue channel remains more discretized than red/green.
