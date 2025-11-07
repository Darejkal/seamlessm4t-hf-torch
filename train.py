import os
import torch
from datasets import load_from_disk
import datasets as ds
from transformers import (
    SeamlessM4TForSpeechToText,
    SeamlessM4TProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    SeamlessM4TConfig,
    GenerationConfig
)
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from evaluate import load
import json
import random
import librosa
from speechbrain.augment.time_domain import (
    SpeedPerturb,
    DropChunk,
    DropFreq,
    AddReverb,
)
import torchaudio
import torchaudio.transforms as T
from soe_vinorm import normalize_text
class AudioAugmentation:
    """Comprehensive audio augmentation class with multiple augmentation techniques"""
    
    def __init__(
        self,
        sample_rate=16000,
        noise_dataset=None,
        # Probabilities for each augmentation
        prob_noise=0.3,
        prob_downsample=0.2,
        prob_spec_augment=0.5,
        prob_speed_perturb=0.3,
        prob_time_dropout=0.2,
        prob_freq_dropout=0.2,
        prob_clipping=0.1,
        prob_random_amp=0.3,
        prob_drop_bit=0.1,
        prob_reverb=0.2,
        # Noise augmentation params
        noise_snr_db_range=(5, 20),
        # Downsample params
        downsample_factors=[2, 4, 5],
        # SpecAugment params
        spec_freq_mask_param=27,
        spec_time_mask_param=100,
        spec_num_freq_masks=2,
        spec_num_time_masks=2,
        # Speed perturbation params
        speed_perturb_factors=[90, 100, 110],  # Percentages, not ratios
        # Time dropout params
        time_drop_length_range=(0.01, 0.05),  # in seconds
        time_drop_count_range=(1, 5),
        # Freq dropout params
        freq_drop_width_range=(0.05, 0.15),  # fraction of frequency bands
        freq_drop_count_range=(1, 3),
        # Clipping params
        clipping_factor_range=(0.3, 0.7),
        # Random amplitude params
        amp_factor_range=(0.5, 1.5),
        # Drop bit resolution params
        bit_depths=[8, 12, 14],
        # Reverb params
        reverb_rir_scale_range=(0.3, 0.8),
        reverb_csv_file_path=None
    ):
        self.sample_rate = sample_rate
        self.noise_dataset = noise_dataset
        
        # Store probabilities
        self.prob_noise = prob_noise
        self.prob_downsample = prob_downsample
        self.prob_spec_augment = prob_spec_augment
        self.prob_speed_perturb = prob_speed_perturb
        self.prob_time_dropout = prob_time_dropout
        self.prob_freq_dropout = prob_freq_dropout
        self.prob_clipping = prob_clipping
        self.prob_random_amp = prob_random_amp
        self.prob_drop_bit = prob_drop_bit
        self.prob_reverb = prob_reverb
        
        # Store parameters
        self.noise_snr_db_range = noise_snr_db_range
        self.downsample_factors = downsample_factors
        self.spec_freq_mask_param = spec_freq_mask_param
        self.spec_time_mask_param = spec_time_mask_param
        self.spec_num_freq_masks = spec_num_freq_masks
        self.spec_num_time_masks = spec_num_time_masks
        self.speed_perturb_factors = speed_perturb_factors
        self.time_drop_length_range = time_drop_length_range
        self.time_drop_count_range = time_drop_count_range
        self.freq_drop_width_range = freq_drop_width_range
        self.freq_drop_count_range = freq_drop_count_range
        self.clipping_factor_range = clipping_factor_range
        self.amp_factor_range = amp_factor_range
        self.bit_depths = bit_depths
        self.reverb_rir_scale_range = reverb_rir_scale_range
        
        # Initialize SpeechBrain augmentations directly
        self._speed_perturb = SpeedPerturb(
            orig_freq=self.sample_rate,
            speeds=self.speed_perturb_factors,
        )
        
        self._drop_chunk = DropChunk(
            drop_length_low=int(self.time_drop_length_range[0] * self.sample_rate),
            drop_length_high=int(self.time_drop_length_range[1] * self.sample_rate),
            drop_count_low=self.time_drop_count_range[0],
            drop_count_high=self.time_drop_count_range[1],
        )
        
        self._drop_freq = DropFreq(
            drop_freq_low=self.freq_drop_width_range[0],
            drop_freq_high=self.freq_drop_width_range[1],
            drop_freq_count_low=self.freq_drop_count_range[0],
            drop_freq_count_high=self.freq_drop_count_range[1],
        )
        if reverb_csv_file_path:
            self._add_reverb = AddReverb(reverb_csv_file_path,reverb_sample_rate=self.sample_rate,clean_sample_rate=self.sample_rate)
        else:
            self._add_reverb=None
    
    def add_noise(self, audio, sr):
        """Mix noise from MUSAN dataset"""
        if self.noise_dataset is None or random.random() > self.prob_noise:
            return audio
        
        try:
            # Get random noise sample
            noise_idx = random.randint(0, len(self.noise_dataset) - 1)
            noise_sample = self.noise_dataset[noise_idx]["audio"]
            noise_audio = noise_sample["array"]
            noise_sr = noise_sample["sampling_rate"]
            
            # Resample noise if needed
            if noise_sr != sr:
                noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=sr)
            
            # Adjust noise length to match audio
            if len(noise_audio) < len(audio):
                # Repeat noise if shorter
                repeats = int(np.ceil(len(audio) / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)[:len(audio)]
            else:
                # Trim noise if longer
                start_idx = random.randint(0, len(noise_audio) - len(audio))
                noise_audio = noise_audio[start_idx:start_idx + len(audio)]
            
            # Calculate SNR and mix
            snr_db = random.uniform(*self.noise_snr_db_range)
            audio_power = np.mean(audio ** 2)
            noise_power = np.mean(noise_audio ** 2)
            snr_linear = 10 ** (snr_db / 10)
            scale = np.sqrt(audio_power / (snr_linear * noise_power + 1e-10))
            
            mixed_audio = audio + scale * noise_audio
            # Normalize to prevent clipping
            max_val = np.abs(mixed_audio).max()
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val
            
            return mixed_audio
        except Exception as e:
            print(f"Warning: Noise augmentation failed: {e}")
            return audio
    
    def downsample_upsample(self, audio, sr):
        """Downsample then upsample back to original sample rate"""
        if random.random() > self.prob_downsample:
            return audio
        
        factor = random.choice(self.downsample_factors)
        target_sr = sr // factor
        
        # Downsample
        downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        # Upsample back
        upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=sr)
        
        return upsampled
    
    def spec_augment_features(self, input_features):
        """Apply SpecAugment to input features (mel spectrogram)"""
        if random.random() > self.prob_spec_augment:
            return input_features
        
        # Convert to tensor if needed
        if isinstance(input_features, np.ndarray):
            features = torch.from_numpy(input_features)
        else:
            features = input_features
        
        # Ensure 3D tensor: [batch, freq, time] or [freq, time]
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        # Apply frequency masking
        for _ in range(self.spec_num_freq_masks):
            freq_mask_param = min(self.spec_freq_mask_param, features.shape[1])
            if freq_mask_param > 0:
                f = random.randint(0, freq_mask_param)
                f0 = random.randint(0, features.shape[1] - f)
                features[:, f0:f0+f, :] = 0
        
        # Apply time masking
        for _ in range(self.spec_num_time_masks):
            time_mask_param = min(self.spec_time_mask_param, features.shape[2])
            if time_mask_param > 0:
                t = random.randint(0, time_mask_param)
                t0 = random.randint(0, features.shape[2] - t)
                features[:, :, t0:t0+t] = 0
        
        # Remove batch dimension if it was added
        if features.shape[0] == 1:
            features = features.squeeze(0)
        
        # Convert back to numpy if input was numpy
        if isinstance(input_features, np.ndarray):
            features = features.numpy()
        
        return features
    
    def speed_perturb(self, audio, sr):
        """Apply speed perturbation using SpeechBrain"""
        if random.random() > self.prob_speed_perturb:
            return audio
        
        try:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            # SpeedPerturb.forward() only takes waveform, no lengths parameter
            augmented = self._speed_perturb(audio_tensor)
            return augmented.squeeze(0).numpy()
        except Exception as e:
            print(f"Warning: Speed perturbation failed: {e}")
            return audio
    
    def time_dropout(self, audio, sr):
        """Apply time dropout (chunk drop)"""
        if random.random() > self.prob_time_dropout:
            return audio
        
        try:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            # DropChunk expects normalized lengths (0-1 range), not absolute sample counts
            lengths = torch.tensor([1.0])  # Full length since we have the complete audio
            augmented = self._drop_chunk(audio_tensor, lengths)
            return augmented.squeeze(0).numpy()
        except Exception as e:
            print(f"Warning: Time dropout failed: {e}")
            return audio
    
    def freq_dropout(self, audio, sr):
        """Apply frequency dropout"""
        if random.random() > self.prob_freq_dropout:
            return audio
        
        try:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            augmented = self._drop_freq(audio_tensor)
            return augmented.squeeze(0).numpy()
        except Exception as e:
            print(f"Warning: Frequency dropout failed: {e}")
            return audio
    
    def clipping(self, audio):
        """Apply random clipping"""
        if random.random() > self.prob_clipping:
            return audio
        
        factor = random.uniform(*self.clipping_factor_range)
        max_val = np.abs(audio).max()
        threshold = max_val * factor
        
        clipped = np.clip(audio, -threshold, threshold)
        # Normalize back
        if np.abs(clipped).max() > 0:
            clipped = clipped / np.abs(clipped).max() * max_val
        
        return clipped
    
    def random_amplitude(self, audio):
        """Apply random amplitude scaling"""
        if random.random() > self.prob_random_amp:
            return audio
        
        factor = random.uniform(*self.amp_factor_range)
        scaled = audio * factor
        
        # Prevent clipping
        max_val = np.abs(scaled).max()
        if max_val > 1.0:
            scaled = scaled / max_val
        
        return scaled
    
    def drop_bit_resolution(self, audio):
        """Reduce bit depth then restore"""
        if random.random() > self.prob_drop_bit:
            return audio
        
        bit_depth = random.choice(self.bit_depths)
        max_val = 2 ** (bit_depth - 1)
        
        # Quantize
        quantized = np.round(audio * max_val) / max_val
        
        return quantized
    
    def add_reverb(self, audio, sr):
        """Add reverberation effect"""
        if not self._add_reverb:
            return audio
        if random.random() > self.prob_reverb:
            return audio
        
        try:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            # AddReverb.forward() only takes waveforms, loads RIR from CSV internally
            augmented = self._add_reverb(audio_tensor)
            return augmented.squeeze(0).numpy()
        except Exception as e:
            print(f"Warning: Reverb augmentation failed: {e}")
            return audio
    
    def augment_audio(self, audio, sr):
        """Apply all audio-level augmentations"""
        # Audio waveform augmentations
        audio = self.add_noise(audio, sr)
        audio = self.downsample_upsample(audio, sr)
        audio = self.speed_perturb(audio, sr)
        audio = self.time_dropout(audio, sr)
        audio = self.freq_dropout(audio, sr)
        audio = self.clipping(audio)
        audio = self.random_amplitude(audio)
        audio = self.drop_bit_resolution(audio)
        audio = self.add_reverb(audio, sr)
        
        return audio
    
    def augment_features(self, input_features):
        """Apply feature-level augmentations (SpecAugment)"""
        return self.spec_augment_features(input_features)


class StatedSeamlessM4TForSpeechToText(SeamlessM4TForSpeechToText):
    def generate(
        self,
        input_features=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        if not tgt_lang:
            tgt_lang=getattr(self.generation_config,"tgt_lang")
        return super().generate(
            input_features,
            tgt_lang,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            **kwargs,
        )
def save_output(pred_output, file):
    """Save prediction output to JSON file"""
    def tolist_safe(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, tuple):
            return [tolist_safe(i) for i in x]
        return x

    pred_dict = {
        "predictions": tolist_safe(pred_output.predictions),
        "label_ids": tolist_safe(pred_output.label_ids),
        "metrics": pred_output.metrics
    }
    with open(file, "w", encoding="utf-8") as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=2)


def main():
    # Set model and processor
    model_name = "facebook/hf-seamless-m4t-medium"
    
    # Load processor and model
    processor = SeamlessM4TProcessor.from_pretrained(model_name)
    processor.tokenizer.tgt_lang="vie"
    # processor.tokenizer.src_lang="khm"
    model = StatedSeamlessM4TForSpeechToText.from_pretrained(model_name)
    model.generation_config.max_new_tokens=4096
    # Configure generation settings
    # For hmong (km) to Vietnamese (vi) translation
    # model.generation_config.forced_decoder_ids = None
    # Set source and target languages if needed
    # model.config.src_lang = "km"  # hmong
    model.generation_config.tgt_lang = "vie"  # Vietnamese

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Custom Dataset to process on-the-fly
    class SeamlessDataset(Dataset):
        def __init__(self, hf_dataset, processor, augmentation:"AudioAugmentation"=None, apply_augmentation=True):
            self.dataset = hf_dataset
            self.processor = processor
            self.augmentation = augmentation
            self.apply_augmentation = apply_augmentation
            # self.src_lang = src_lang
            # self.tgt_lang = tgt_lang
        
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            audio = item["audio"]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            
            # Apply audio augmentations if enabled
            if self.apply_augmentation and self.augmentation is not None:
                audio_array = self.augmentation.augment_audio(audio_array, sampling_rate)
            
            # Process audio input
            inputs = self.processor(
                audio=audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                # src_lang=self.src_lang,
            )
            input_features = inputs.input_features[0]
            
            # Apply feature augmentations (SpecAugment) if enabled
            if self.apply_augmentation and self.augmentation is not None:
                input_features = self.augmentation.augment_features(input_features)
            
            # Process text labels
            # Assuming your dataset has a "text" field for Vietnamese text
            labels = self.processor.tokenizer(
                text_target=normalize_text(item["vi"].lower().strip()),
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096
            ).input_ids[0]
            
            return {
                "input_features": input_features,
                "labels": labels
            }

    # Load your dataset (replace with your own data path)
    print("Loading dataset...")
    raw_dataset = ds.load_dataset("Darejkal/bana-record-20251104")
    raw_dataset=  raw_dataset.filter(lambda x:x['device'].strip().lower()!="True Coffee".lower(),num_proc=40)
    raw_dataset=raw_dataset.cast_column("audio",ds.Audio(sampling_rate=16000))
    train_dataset = SeamlessDataset(raw_dataset['train'], processor)
    eval_dataset = train_dataset
    # eval_dataset = SeamlessDataset(raw_dataset['test'], processor)
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./ss-bahnar-test-20251104",
        per_device_train_batch_size=4,  # Smaller batch size for larger model
        do_eval=False,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,  # Save memory
        # eval_strategy="epoch",
        num_train_epochs=100,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=1e-5,  # Lower learning rate for finetuning
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        push_to_hub=True,
        hub_model_id="Darejkal/ss-bahnar-test-20251104",
        hub_strategy="every_save",
        hub_token=os.environ.get("HF_TOKEN"),
        hub_private_repo=True,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,  # Important for custom datasets
        # load_best_model_at_end=True,
        # metric_for_best_model="wer",
        # greater_is_better=False,  # Lower WER is better
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=448,
        ddp_find_unused_parameters=True,
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Extract input features and labels
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            
            # Pad input features
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            # Pad labels
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # Remove BOS token if present (will be added by model)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            
            return batch

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Load WER metric
    wer_metric = load("wer")

    def compute_metrics(pred):
        """Compute WER metric"""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}

    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate before training (optional)
    print("Evaluating before training...")
    if training_args.do_eval:
        pred_output = trainer.predict(eval_dataset)
        if trainer.is_world_process_zero():
            save_output(pred_output, "seamless_prediction_output_hmong_before.json")
            print(f"WER before training: {pred_output.metrics['test_wer']:.4f}")
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=False)
    
    # Save final model
    if trainer.is_world_process_zero():
        print("Saving final model...")
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
    
    # Evaluate after training
    print("Evaluating after training...")
    if training_args.do_eval:
        pred_output = trainer.predict(eval_dataset)
        if trainer.is_world_process_zero():
            save_output(pred_output, "seamless_prediction_output_hmong_after.json")
            print(f"WER after training: {pred_output.metrics['test_wer']:.4f}")


if __name__ == "__main__":
    main()
