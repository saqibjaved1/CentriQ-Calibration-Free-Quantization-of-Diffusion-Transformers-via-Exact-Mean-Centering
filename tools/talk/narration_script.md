# CentriQ intro video: narration script

## Slide 1

Hi, I'm going to present CentriQ, a way to quantize diffusion transformers to 4-bit weights and 4-bit activations without any calibration data.

## Slide 2

Quantizing both weights and activations to 4 bits is what actually speeds up a diffusion transformer, because its forward pass is compute-bound. But every method that reaches W4A4 on diffusion models today relies on calibration: smoothing factors, clipping ranges, rotations or low-rank corrections, fitted to prompts run through one specific checkpoint. For FLUX, SVDQuant's calibration took us thirteen hours, and it has to be redone for every model, fine-tune and bit-width.

## Slide 3

Without calibration, things fall apart. Plain round-to-nearest and HQQ produce pure noise on PixArt-Sigma, with an FID of 481, and lose the content of the prompt on FLUX and SANA. On the right you can see CentriQ, which stays close to the 16-bit output on all three models without seeing any data. So what breaks?

## Slide 4

Diffusion transformers condition every block through adaptive layer norm, which shifts each token by an offset. So every activation vector carries a per-token mean. The standard data-free fix for outliers is a Hadamard rotation, but a rotation preserves energy, and it maps this mean onto one or a few coordinates. That spike, shown here in blue, then sets the quantization step for the entire token. Remove it, and the 4-bit grid for this token becomes 2.4 times finer.

## Slide 5

Could a better rotation spread the mean evenly? Only a regular Hadamard matrix, one with equal row sums, does that, and those exist only at orders four u squared. None of the evaluated widths, 1152, 2240 or 3072, has that form. On the right: rotation alone leaves the per-token max-to-RMS ratio where it was, while centering before rotation brings it down to what incoherence predicts.

## Slide 6

CentriQ does three things. It subtracts each token's mean, which costs order n and is computed in the same kernel launch as the scale. It rotates the centered token with a fixed factorized Hadamard matrix and quantizes it with one closed-form scale per token. And because the layer is linear, it adds the mean back exactly, as a rank-1 term, the mean times W times one, in full precision inside the GEMM epilogue. The mean never touches a quantizer.

## Slide 7

Here is one quantized layer. The online path centers, pads and rotates each token, quantizes it to INT4 and runs the INT4 GEMM. The orange branch carries the mean around both quantizers. Offline, the weights are fitted under a robust l-p objective with p equal to 0.7, which we call RoDS. Because the marginal cost of a large residual shrinks, the dense mode of each weight group sets the scale, not its heavy tail. Neither path uses data.

## Slide 8

Here are the main results at W4A4 on MJHQ. Across PixArt-Sigma, FLUX and SANA, CentriQ matches calibrated SVDQuant on FID and ImageReward and has the lowest LPIPS against the 16-bit images, without using any data. Because SVDQuant's numbers come from its paper, we also regenerated it under our exact protocol: the two match on FLUX, and CentriQ is ahead on PixArt-Sigma.

## Slide 9

Here is what that looks like. Every image in a row shares the prompt and the initial noise. On PixArt-Sigma, CentriQ stays close to the 16-bit wolf, while HadaNorm-style centering and rotation alone, run in the same setting, break the image. On FLUX, CentriQ and calibrated SVDQuant are both close to the 16-bit image.

## Slide 10

Speed. Both 4-bit methods run in the same fused Nunchaku engine, so they differ only in the quantizer. CentriQ is within 3.5 percent of SVDQuant's latency on an A100, and within 2.8 percent on an RTX PRO 6000, where its transformer is 1.76 times faster than eager BF16. Converting a new checkpoint takes 9.3 minutes for FLUX, against thirteen hours of calibration, with about the same peak memory.

## Slide 11

Below four bits. On FLUX, CentriQ is the first calibration-free method to keep usable image quality at 2-bit activations, and it outperforms the strongest calibration-free method reported so far at 2-bit weights. The ablation on the right shows why: rotation helps, centering closes the gap at four bits, and a single shared clip percentile brings W4A2 from an FID of 444 down to 36.

## Slide 12

To sum up: the per-token mean that adaptive layer norm adds is what breaks data-free rotations on diffusion transformers. CentriQ centers each token, rotates, and restores the mean exactly, so it never touches a quantizer. With no calibration data, it matches calibrated SVDQuant at W4A4 on three models, runs within 3.5 percent of its latency, and is the first calibration-free method with usable quality at 2-bit activations. The paper is on arXiv, and the code and project page are coming soon. Thanks for watching.

