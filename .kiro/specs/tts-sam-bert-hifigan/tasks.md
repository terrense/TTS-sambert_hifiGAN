# Implementation Plan

- [x] 1. Set up project structure and core infrastructure





  - Create directory structure: configs/, models/, data/, scripts/, tests/
  - Create requirements.txt with all dependencies (PyTorch 2.x, torchaudio, numpy, pyyaml, pytest)
  - Create main config.yaml with audio parameters (sample_rate, n_fft, hop_length, win_length, n_mels, fmin, fmax)
  - Create model_config.yaml with model hyperparameters
  - _Requirements: 21.1, 21.2, 21.3, 22.1, 22.2, 22.3, 22.5_

- [x] 2. Implement audio processing utilities





  - Create data/audio_processing.py with mel extraction function using torchaudio
  - Implement extract_mel() function that takes waveform and returns log-mel spectrogram [n_mels, T]
  - Ensure mel parameters are loaded from config.yaml
  - Add shape logging for mel extraction
  - _Requirements: 15.2, 15.3_

- [x] 3. Implement Front-end text processing module





  - Create models/frontend.py with FrontEnd class
  - Implement simple character-to-token mapping (pseudo G2P)
  - Output LinguisticFeature dataclass with ph_ids [B, Tph], tone_ids [B, Tph], boundary_ids [B, Tph]
  - Add shape logging in forward pass
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 4. Implement Phoneme Embedding module





  - Create models/phoneme_embedding.py with PhonemeEmbedding class
  - Implement three embedding layers for ph_ids, tone_ids, boundary_ids
  - Sum embeddings to produce H0 [B, Tph, d_model]
  - Add shape logging for input and output tensors
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Implement BERT Encoder





  - Create models/bert_encoder.py with BERTEncoder class
  - Implement multi-layer Transformer Encoder using nn.TransformerEncoderLayer
  - Support configurable n_layers, n_heads, d_model, d_ff, dropout
  - Input H0 [B, Tph, d], output Henc [B, Tph, d]
  - Add shape logging for input and output
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6. Implement Duration Predictor





  - Create models/variance_adaptor.py with DurationPredictor class
  - Use Conv1d layers with ReLU and LayerNorm
  - Input Henc [B, Tph, d], output log_dur_pred [B, Tph]
  - Add shape logging
  - _Requirements: 4.1, 4.3_

- [x] 7. Implement Length Regulator













  - Add LengthRegulator class to models/variance_adaptor.py
  - Implement repeat logic using torch.repeat_interleave based on duration
  - Input Henc [B, Tph, d] and dur [B, Tph], output Hlr [B, Tfrm, d]
  - Add shape logging
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Implement Pitch Predictor





  - Add PitchPredictor class to models/variance_adaptor.py
  - Reuse DurationPredictor architecture for prediction
  - Implement pitch quantization with n_bins and embedding layer
  - Expand phoneme-level to frame-level using duration
  - Output pitch_tok [B, Tph], pitch_frm [B, Tfrm], Ep [B, Tfrm, d]
  - Add shape logging for all intermediate tensors
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 9. Implement Energy Predictor





  - Add EnergyPredictor class to models/variance_adaptor.py
  - Similar architecture to PitchPredictor
  - Output energy_tok [B, Tph], energy_frm [B, Tfrm], Ee [B, Tfrm, d]
  - Add shape logging
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 10. Integrate Variance Adaptor





  - Add VarianceAdaptor class combining all predictors
  - Implement forward pass: duration prediction -> length regulation -> pitch/energy prediction
  - Compute Hvar = Hlr + Ep + Ee
  - Return Hvar [B, Tfrm, d] and all predictions dict
  - Add shape logging for Hvar
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 11. Implement PNCA AR-Decoder with teacher forcing




  - Create models/ar_decoder.py with PNCAARDecoder class
  - Implement prenet (Linear -> ReLU -> Dropout -> Linear)
  - Implement multi-layer Transformer Decoder using nn.TransformerDecoderLayer
  - Implement training mode with teacher forcing: shift mel_gt and use as decoder input
  - Input Hvar [B, Tfrm, d] and mel_gt [B, Tfrm, n_mels], output mel_pred [B, Tfrm, n_mels]
  - Add shape logging
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 12. Implement AR-Decoder autoregressive inference
  - Add inference mode to PNCAARDecoder
  - Implement chunk-based autoregressive generation (generate C frames at a time)
  - Support max_len parameter for generation length
  - Add shape logging during generation
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 13. Implement complete SAM-BERT Acoustic Model
  - Create models/acoustic_model.py with SAMBERTAcousticModel class
  - Integrate PhonemeEmbedding, BERTEncoder, VarianceAdaptor, PNCAARDecoder
  - Implement forward pass connecting all modules
  - Add shape logging at each stage (H0, Henc, Hvar, mel_pred)
  - Return mel_pred and predictions dict
  - _Requirements: 2.1, 3.1, 8.1, 9.1_

- [ ] 14. Implement acoustic model loss functions
  - Create models/losses.py with AcousticLoss class
  - Implement L_mel = L1(mel_pred, mel_gt)
  - Implement L_dur = MSE(log_dur_pred, log(dur_gt+1))
  - Implement L_pitch = MSE(pitch_pred, pitch_gt) with mask
  - Implement L_energy = MSE(energy_pred, energy_gt)
  - Return total loss and loss_dict with all components
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 4.4, 6.5, 7.5_

- [ ] 15. Implement HiFi-GAN Generator
  - Create models/hifigan.py with HiFiGANGenerator class
  - Implement Conv-Pre (1D conv to project mel to hidden dim)
  - Implement Upsample blocks using ConvTranspose1d
  - Implement Multi-Receptive Field (MRF) residual blocks
  - Implement Conv-Post (final projection to waveform)
  - Input mel [B, n_mels, Tfrm], output wav [B, 1, T_wav] where T_wav = Tfrm * hop_length
  - Add shape logging
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [ ] 16. Implement HiFi-GAN Multi-Scale Discriminator
  - Add MultiScaleDiscriminator class to models/hifigan.py
  - Implement 3 discriminators with different downsampling rates (1x, 2x, 4x)
  - Use AvgPool1d for downsampling
  - Return discriminator outputs and intermediate feature maps
  - Add shape logging
  - _Requirements: 13.1, 13.3, 13.4_

- [ ] 17. Implement HiFi-GAN Multi-Period Discriminator
  - Add MultiPeriodDiscriminator class to models/hifigan.py
  - Implement discriminators for periods [2, 3, 5, 7, 11]
  - Reshape waveform based on period before processing
  - Return discriminator outputs and feature maps
  - Add shape logging
  - _Requirements: 13.2, 13.3, 13.4_

- [ ] 18. Implement HiFi-GAN loss functions
  - Add VocoderLoss class to models/losses.py
  - Implement adversarial loss (hinge loss or least squares)
  - Implement feature matching loss between real and fake feature maps
  - Implement multi-resolution STFT loss
  - Return generator loss, discriminator loss, and loss_dict
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 19. Integrate complete HiFi-GAN model
  - Add HiFiGAN class to models/hifigan.py
  - Integrate Generator, MSD, and MPD
  - Implement forward() for generation: mel -> wav
  - Implement discriminate() for training: process real and fake waveforms
  - Add shape logging
  - _Requirements: 12.1, 13.1, 13.2_

- [ ] 20. Create shape validation tests
  - Create tests/test_shapes.py
  - Test FrontEnd output shapes
  - Test PhonemeEmbedding output shape [B, Tph, d]
  - Test BERTEncoder output shape [B, Tph, d]
  - Test VarianceAdaptor output shape [B, Tfrm, d]
  - Test AR-Decoder output shape [B, Tfrm, n_mels]
  - Test HiFi-GAN Generator output shape [B, 1, T_wav]
  - Use random inputs and assert all shapes are correct
  - _Requirements: 19.1, 19.2, 19.3, 19.4_

- [ ] 21. Create end-to-end inference script
  - Create scripts/inference.py
  - Implement text_to_wav() function
  - Load configs and initialize all models (FrontEnd, SAM-BERT, HiFi-GAN)
  - Implement pipeline: text -> linguistic features -> mel -> wav
  - Save output wav file using torchaudio.save()
  - Add shape logging at each stage
  - _Requirements: 17.1, 17.2, 17.3, 17.4_

- [ ] 22. Create streaming inference demo
  - Create scripts/streaming_demo.py
  - Implement chunk-based mel generation using AR-Decoder
  - Implement StreamingBuffer class for overlap-add/crossfade
  - Generate wav chunks and concatenate with smooth transitions
  - Save final wav file
  - Add shape logging for each chunk
  - _Requirements: 18.1, 18.2, 18.3, 18.4_

- [ ] 23. Create end-to-end inference test
  - Create tests/test_infer.py
  - Implement test_text_to_wav() that runs complete pipeline
  - Use sample text input "你好世界"
  - Verify output wav file exists and has correct format
  - Verify sample rate and channel count
  - _Requirements: 20.1, 20.2, 20.3, 20.4_

- [ ] 24. Create shape contract documentation
  - Create docs/shape_contract.md
  - Document all module input/output shapes with clear notation
  - Include Front-end, Phoneme Embedding, BERT Encoder, Variance Adaptor, AR-Decoder, HiFi-GAN
  - Use symbols: B (batch), Tph (phoneme length), Tfrm (frame length), T_wav (waveform length), d (hidden dim), n_mels
  - _Requirements: 16.1, 16.2, 16.3, 16.4_

- [ ] 25. Create acoustic model training script
  - Create scripts/train_acoustic.py
  - Implement training loop with DataLoader
  - Load SAM-BERT model and AcousticLoss
  - Implement optimizer (AdamW), learning rate scheduler
  - Add checkpointing and logging (TensorBoard)
  - Log all loss components during training
  - _Requirements: 11.5, 15.1_

- [ ] 26. Create vocoder training script
  - Create scripts/train_vocoder.py
  - Implement GAN training loop (alternate generator and discriminator)
  - Load HiFi-GAN model and VocoderLoss
  - Implement separate optimizers for generator and discriminators
  - Add checkpointing and logging
  - _Requirements: 14.4, 15.1_

- [ ] 27. Create dataset implementation
  - Create data/dataset.py with TTSDataset class
  - Implement __getitem__ to load wav, extract mel, load duration/pitch/energy
  - Support metadata.csv format
  - Implement collate_fn for batching with padding
  - _Requirements: 15.1_

- [ ] 28. Add mel extraction consistency test
  - Add test_mel_consistency() to tests/test_shapes.py
  - Load test audio and extract mel
  - Verify mel shape [n_mels, T]
  - Verify mel value range (log-mel should be negative)
  - _Requirements: 15.3_

- [ ] 29. Add configuration validation
  - Create utils/config_validation.py
  - Implement validate_mel_config() to check consistency between acoustic and vocoder configs
  - Verify n_mels, hop_length, sample_rate match
  - Raise clear error messages if mismatch detected
  - _Requirements: 15.3_

- [ ] 30. Add mixed precision training support
  - Update training scripts to use torch.cuda.amp
  - Implement GradScaler for automatic mixed precision
  - Add config option to enable/disable AMP
  - _Requirements: 22.1, 22.2_
