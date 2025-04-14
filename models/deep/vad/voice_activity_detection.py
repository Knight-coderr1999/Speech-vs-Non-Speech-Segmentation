from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token="")


vad_pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  "onset": 0.5, "offset": 0.5,
  "min_duration_on": 0.0,
  "min_duration_off": 0.0
}

vad_pipeline.instantiate(HYPER_PARAMETERS)
def vad_segmentation(input_path, output_path, aggressiveness=2):
    diarization = vad_pipeline(input_path)
    with open(output_path, "w") as f:
        diarization.write_rttm(f)
