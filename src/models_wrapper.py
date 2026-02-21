import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC, WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC

class BaseASRModel:
    def load_audio(self, audio_path):
        # Tous les modèles HF s'attendent généralement à du 16kHz
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()

    def post_process(self, text):
        # Standardisation de la sortie : On enlève les espaces, ponctuation, et on met en majuscule
        # Comme on attend une lettre, on extrait le premier caractère alphanumérique trouvé
        clean_text = ''.join(e for e in text if e.isalnum()).upper()
        return clean_text[0] if len(clean_text) > 0 else ""

class Wav2Vec2Wrapper(BaseASRModel):
    def __init__(self, config):
        # Utilisation explicite des classes Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(config["hf_path"])
        self.model = Wav2Vec2ForCTC.from_pretrained(config["hf_path"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, audio_path):
        audio_input = self.load_audio(audio_path)
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        # Argmax sur la dernière dimension (le vocabulaire)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Décodage via le processeur spécifique
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return self.post_process(transcription)

class WhisperWrapper(BaseASRModel):
    def __init__(self, config):
        self.processor = WhisperProcessor.from_pretrained(config["hf_path"])
        self.model = WhisperForConditionalGeneration.from_pretrained(config["hf_path"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.language = config.get("language", "fr")

    def predict(self, audio_path):
        audio_input = self.load_audio(audio_path)
        input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        # Forcer la langue française
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return self.post_process(transcription)

def get_model(model_config):
    if model_config["type"] == "whisper":
        return WhisperWrapper(model_config)
    elif model_config["type"] == "wav2vec2":
        return Wav2Vec2Wrapper(model_config)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_config['type']}")