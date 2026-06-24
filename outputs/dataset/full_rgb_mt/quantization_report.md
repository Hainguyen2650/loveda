# Blue Channel Quantization Check

This report compares `R`, `G`, and `B` channel histograms using simple quantization-oriented diagnostics:
- histogram roughness via first and second differences,
- even-vs-odd parity bias,
- residue bias under `mod 2`, `mod 4`, and `mod 8`,
- and average unique intensity count per image.

Interpretation rule:
- If `B` consistently has higher roughness and stronger modulo residue bias than `R/G`, that supports the quantization/discretization hypothesis.

## Train Urban

- `R`: roughness-1=0.0001243285, roughness-2=0.0000132378, parity-bias=0.0003505386, midrange peak/trough=2.739760, mod8 dominant residue=7, mod8 uniformity-l1=0.0005277058
- `G`: roughness-1=0.0001465722, roughness-2=0.0000165434, parity-bias=0.0001364032, midrange peak/trough=6.366220, mod8 dominant residue=7, mod8 uniformity-l1=0.0020783809
- `B`: roughness-1=0.0007620522, roughness-2=0.0013837686, parity-bias=0.0002401644, midrange peak/trough=14.831094, mod8 dominant residue=3, mod8 uniformity-l1=0.1220038030

## Train Rural

- `R`: roughness-1=0.0001067624, roughness-2=0.0000114269, parity-bias=0.0004000986, midrange peak/trough=2.269828, mod8 dominant residue=0, mod8 uniformity-l1=0.0017555325
- `G`: roughness-1=0.0001212340, roughness-2=0.0000097906, parity-bias=0.0001976501, midrange peak/trough=3.117213, mod8 dominant residue=7, mod8 uniformity-l1=0.0009452951
- `B`: roughness-1=0.0003844390, roughness-2=0.0006114428, parity-bias=0.0002108859, midrange peak/trough=3.788951, mod8 dominant residue=3, mod8 uniformity-l1=0.0611040502

## Val Urban

- `R`: roughness-1=0.0001404994, roughness-2=0.0000149320, parity-bias=0.0001388102, midrange peak/trough=3.105744, mod8 dominant residue=7, mod8 uniformity-l1=0.0006455748
- `G`: roughness-1=0.0001609521, roughness-2=0.0000152435, parity-bias=0.0002544604, midrange peak/trough=4.496551, mod8 dominant residue=7, mod8 uniformity-l1=0.0004168067
- `B`: roughness-1=0.0007351744, roughness-2=0.0013305394, parity-bias=0.0004647548, midrange peak/trough=10.349144, mod8 dominant residue=3, mod8 uniformity-l1=0.1145113379

## Val Rural

- `R`: roughness-1=0.0001111246, roughness-2=0.0000153095, parity-bias=0.0007876811, midrange peak/trough=3.484017, mod8 dominant residue=0, mod8 uniformity-l1=0.0018787731
- `G`: roughness-1=0.0001172155, roughness-2=0.0000114736, parity-bias=0.0001752716, midrange peak/trough=2.248487, mod8 dominant residue=0, mod8 uniformity-l1=0.0008605207
- `B`: roughness-1=0.0002868685, roughness-2=0.0004162195, parity-bias=0.0005084379, midrange peak/trough=4.803781, mod8 dominant residue=3, mod8 uniformity-l1=0.0377526782

## Test Urban

- `R`: roughness-1=0.0001808374, roughness-2=0.0000328993, parity-bias=0.0006218384, midrange peak/trough=5.270032, mod8 dominant residue=7, mod8 uniformity-l1=0.0054803105
- `G`: roughness-1=0.0001701847, roughness-2=0.0000303580, parity-bias=0.0007448470, midrange peak/trough=9.205580, mod8 dominant residue=5, mod8 uniformity-l1=0.0042447706
- `B`: roughness-1=0.0007120245, roughness-2=0.0012532593, parity-bias=0.0005799569, midrange peak/trough=9.673449, mod8 dominant residue=2, mod8 uniformity-l1=0.1126133405

## Test Rural

- `R`: roughness-1=0.0001428608, roughness-2=0.0000169434, parity-bias=0.0008525003, midrange peak/trough=3.053576, mod8 dominant residue=7, mod8 uniformity-l1=0.0015448504
- `G`: roughness-1=0.0001503304, roughness-2=0.0000144983, parity-bias=0.0008058616, midrange peak/trough=4.494830, mod8 dominant residue=7, mod8 uniformity-l1=0.0013115820
- `B`: roughness-1=0.0006835580, roughness-2=0.0012080367, parity-bias=0.0009421254, midrange peak/trough=10.809844, mod8 dominant residue=3, mod8 uniformity-l1=0.1128165291

Recommendation:
- Treat the blue-channel histogram as a dataset property if its roughness and modulo bias remain systematically stronger after reruns.
- Keep normalization explicit and use moderate color augmentation if the blue channel remains more discretized than red/green.
