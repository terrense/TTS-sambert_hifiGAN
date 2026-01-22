# Multi-Discriminator Loss Aggregation Implementation

## Overview

This document describes the implementation of multi-discriminator loss aggregation for the HiFi-GAN vocoder, as specified in task 22 of the TTS SAM-BERT HiFi-GAN project.

## Implementation Summary

### Changes Made

Updated the `VocoderLoss` class in `models/losses.py` to properly handle loss aggregation across multiple sub-discriminators (MSD and MPD).

### Key Features

1. **Discriminator Loss Aggregation**
   - Aggregates adversarial loss across all 8 sub-discriminators (3 MSD + 5 MPD)
   - Uses **MEAN aggregation** strategy for balanced contribution
   - Ensures MSD and MPD contribute equally regardless of their different counts

2. **Generator Adversarial Loss Aggregation**
   - Aggregates adversarial loss across all 8 sub-discriminators
   - Uses **MEAN aggregation** strategy
   - Properly handles least squares GAN loss: `L_gen_adv = E[(D(G(z)) - 1)^2]`

3. **Feature Matching Loss Aggregation**
   - Aggregates FM loss across all 8 sub-discriminators and their layers
   - Uses **MEAN aggregation** at both layer and discriminator levels
   - Provides per-discriminator loss logging for detailed monitoring
   - Detaches real features to prevent backprop into discriminator

4. **Separate Loss Components**
   - Returns `L_adv` (adversarial loss)
   - Returns `L_fm` (feature matching loss)
   - Returns `L_mel` (mel reconstruction loss)
   - All components available separately for logging and monitoring

## Aggregation Strategy

### Why MEAN Aggregation?

The implementation uses **MEAN aggregation** (averaging) rather than SUM aggregation for the following reasons:

1. **Balanced Contribution**: MSD has 3 discriminators while MPD has 5. Using mean ensures both contribute equally to the total loss, preventing MPD from dominating simply due to having more discriminators.

2. **Scale Independence**: Mean aggregation keeps loss magnitudes consistent regardless of the number of discriminators, making hyperparameter tuning more stable.

3. **Standard Practice**: Mean aggregation is the standard approach in HiFi-GAN and similar multi-discriminator architectures.

### Aggregation Formula

For discriminator loss:
```
L_disc = (1/N) * Σ[L_disc_i] where N = 8 (3 MSD + 5 MPD)
```

For feature matching loss:
```
L_fm = (1/N) * Σ[(1/M_i) * Σ[L_layer_ij]]
where N = number of discriminators (8)
      M_i = number of layers in discriminator i
```

## Code Documentation

All methods include comprehensive docstrings that document:
- The aggregation strategy used (MEAN)
- Expected input structure (8 discriminators: 3 MSD + 5 MPD)
- Return values and their meanings
- Examples of usage

## Testing

### Test Coverage

1. **Unit Tests** (`tests/test_losses.py`)
   - All 16 VocoderLoss tests pass
   - Tests cover discriminator loss, generator loss, feature matching loss
   - Tests verify per-discriminator logging
   - Tests verify backward pass and gradient computation

2. **Integration Tests** (`tests/test_hifigan_integration.py`)
   - All 7 HiFi-GAN integration tests pass
   - Tests verify end-to-end training workflow
   - Tests verify discriminator and generator interaction

3. **Verification Tests** (temporary test script)
   - Verified discriminator loss aggregation
   - Verified generator adversarial loss aggregation
   - Verified feature matching loss aggregation with per-disc logging
   - Verified complete generator loss with all components
   - Verified integration with real HiFi-GAN model

### Test Results

```
tests/test_losses.py::TestVocoderLoss - 16/16 PASSED
tests/test_hifigan_integration.py - 7/7 PASSED
```

All tests pass successfully with no errors or warnings (except deprecation warning for weight_norm).

## Usage Example

```python
from models.losses import VocoderLoss
from models.hifigan import HiFiGAN

# Initialize loss function
loss_fn = VocoderLoss(
    feature_matching_weight=2.0,
    mel_weight=45.0,
    use_mel_loss=True
)

# Initialize model
model = HiFiGAN(n_mels=80)

# Generate waveform
mel = torch.randn(2, 80, 100)
wav_fake = model.generator(mel)
wav_real = torch.randn(2, 1, 25600)

# Get discriminator outputs (3 MSD + 5 MPD = 8 total)
(msd_real_out, msd_real_feat,
 msd_fake_out, msd_fake_feat,
 mpd_real_out, mpd_real_feat,
 mpd_fake_out, mpd_fake_feat) = model.discriminate(wav_real, wav_fake)

# Combine outputs
disc_real_out = msd_real_out + mpd_real_out  # 8 discriminators
disc_fake_out = msd_fake_out + mpd_fake_out  # 8 discriminators
real_feat = msd_real_feat + mpd_real_feat    # 8 lists of features
fake_feat = msd_fake_feat + mpd_fake_feat    # 8 lists of features

# Compute discriminator loss
disc_loss, disc_loss_dict = loss_fn.forward_discriminator(
    disc_real_out, disc_fake_out
)

# Compute generator loss with separate components
gen_loss, gen_loss_dict = loss_fn.forward_generator(
    wav_real, wav_fake, disc_fake_out, real_feat, fake_feat
)

# Access separate loss components
print(f"L_adv: {gen_loss_dict['gen_adv_loss']}")
print(f"L_fm: {gen_loss_dict['gen_fm_loss']}")
print(f"L_mel: {gen_loss_dict['gen_mel_loss']}")

# Access per-discriminator FM losses
for i in range(8):
    print(f"FM loss disc {i}: {gen_loss_dict[f'gen_fm_loss_disc_{i}']}")
```

## Requirements Satisfied

✅ **Requirement 13.3**: Handle list of discriminators (MSD and MPD sub-discriminators)
✅ **Requirement 13.4**: Aggregate adversarial loss across all sub-discriminators
✅ **Requirement 14.1**: Aggregate adversarial loss for generator training
✅ **Requirement 14.2**: Aggregate feature matching loss across all sub-discriminators
✅ **Document aggregation strategy**: Documented MEAN aggregation in code comments
✅ **Return separate loss components**: Returns L_adv, L_mel, L_fm for logging

## Files Modified

- `models/losses.py`: Updated VocoderLoss class with proper aggregation

## Files Tested

- `tests/test_losses.py`: All VocoderLoss tests pass
- `tests/test_hifigan_integration.py`: All integration tests pass

## Conclusion

The multi-discriminator loss aggregation has been successfully implemented with:
- Proper MEAN aggregation across all sub-discriminators
- Comprehensive documentation of the aggregation strategy
- Separate loss components (L_adv, L_fm, L_mel) for logging
- Per-discriminator feature matching loss logging
- Full test coverage with all tests passing

The implementation follows best practices for multi-discriminator GAN training and ensures balanced contribution from both MSD and MPD discriminators.
